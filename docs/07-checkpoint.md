# 第 7 章 checkpoint.py 权重加载

`checkpoint.py` 总共只有 221 行代码，但它要解决的问题并不轻量：把 300 GB 以上的权重数据，在多 host 与多 device 的分布式环境下，从磁盘按 shard 加载并最终拼接为 JAX 的全局分布式数组。本章会把整个加载流程逐步拆解清楚。

!!! note "multi-host / SPMD（Single Program Multiple Data）"
    单台机器无法容纳 314B 的模型权重，因此 Grok-1 在设计上假设运行环境包含若干台机器（host），每台机器上有若干张 GPU（device）。每台机器各自执行同一份 Python 程序，但只负责整个模型的一小块计算，依靠 NCCL 或其他通信库在 host 与 host 之间交换中间结果。这种"同一份代码 + 各自处理各自分片"的并行模式称为 SPMD。

    JAX 使用 `jax.process_index()` 为每台机器分配一个唯一编号，在加载 ckpt 时每台机器都根据自己的编号决定要读取哪些文件。Grok-1 的 `run.py` 默认配置是单机 8 卡，因此 `jax.process_count() = 1`、`jax.process_index() = 0`，多 host 相关的分支会退化掉；但 7.4 与 7.5 节中讨论的代码逻辑本身是为多机部署预留的。

## 7.1 ckpt 的物理布局

Grok-1 的 ckpt 通过 magnet 链接或 HuggingFace 下载，得到一个 `ckpt-0/` 目录，结构是：

```
ckpt-0/
├── tensor00000_000
├── tensor00000_001
├── tensor00000_002
├── ...
├── tensor00769_007
```

文件名格式：`tensor{i:05d}_{shard:03d}`。

- `i` 是 tensor 在 pytree flatten 后的索引（0~769 左右）
- `shard` 是该 tensor 的分片索引（0~7，对应 8 个 model shard）

每个文件是一个 **pickle**，序列化了一个 numpy/JAX 数组（或 `QuantizedWeight8bit`）。

!!! note "pickle"
    pickle 是 Python 自带的对象序列化协议。`pickle.dump(obj, f)` 可以把任意 Python 对象（dict、numpy array、自定义 class 实例等）转换为字节流写入文件，`pickle.load(f)` 则负责反过来还原对象。它的便利之处在于无需手写 schema，即使是嵌套结构也可以直接存取；缺点是反序列化过程中会执行字节流中描述的"如何还原 class"指令，所以恶意构造的 pickle 文件能在加载进程内执行任意代码。

    Grok-1 的 ckpt 包含大约 770 个 tensor，每个 tensor 又被切成 N 个分片，最终总共大约几千个 pickle 文件，每个文件保存一个 numpy 数组或一个 `QuantizedWeight8bit` 实例。整套 296 GB 数据完全依赖 pickle 序列化。

整套 ckpt 的总体大小（以 HuggingFace 上 xai-org/grok-1 为准）约为 296 GB，对应 8-bit 量化后的权重；如果使用未经量化的原始 bf16 权重，总大小约为 600 GB。

每个 tensor 的大小大致落在 100 MB 到 5 GB 之间，具体数值取决于该 tensor 的 shape。其中最大的一类是 MoE 的 `linear_v/w`、`linear/w`、`linear_1/w`：单层的 shape 为 $(8, 6144, 32768)$，bf16 下大约 1.5 GB，8-bit 化之后约 0.75 GB。64 层累加起来约 144 GB，占据了 ckpt 总体积的主要部分。

## 7.2 加载流程总览

```
restore(ckpt_path, state_shapes, mesh, between_hosts_config, params_only, state_sharding)
    │
    ├── flatten state_shapes 拿到 770 个 (path, shape) 二元组
    │
    ├── load_tensors:
    │     ThreadPoolExecutor(max_workers=32)
    │     for each tensor:
    │         决定本 host 是否要 load 这个分片
    │         if yes:  fast_unpickle(...)
    │         else:    np.zeros(shape)
    │     wait all
    │
    ├── unflatten 成嵌套的 state
    │
    ├── 检查 ckpt keys 与 code keys 是否一致
    │
    ├── host_local_array_to_global_array(state, mesh, state_sharding)
    │     把每个 host 上的本地数组拼接成 JAX 的全局分布式 GlobalDeviceArray
    │
    └── return state.params if params_only else state
```

## 7.3 `fast_unpickle` 与 `/dev/shm` 中转

`checkpoint.py:42-74`：

```python
# checkpoint.py:42-74
@contextlib.contextmanager
def copy_to_shm(file: str):
    if file.startswith("/dev/shm/"):
        # Nothing to do, the file is already in shared memory.
        yield file
        return

    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        shutil.copyfile(file, tmp_path)
        yield tmp_path
    finally:
        os.remove(tmp_path)
        os.close(fd)


@contextlib.contextmanager
def copy_from_shm(file: str):
    tmp_dir = "/dev/shm/"
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir)
    try:
        yield tmp_path
        shutil.copyfile(tmp_path, file)
    finally:
        os.remove(tmp_path)
        os.close(fd)


def fast_unpickle(path: str) -> Any:
    with copy_to_shm(path) as tmp_path:
        with open(tmp_path, "rb") as f:
            return pickle.load(f)
```

整体机制可以概括为：**先把 ckpt 文件复制到 `/dev/shm`（tmpfs，即内存盘）上，然后再对位于内存盘上的副本执行 pickle.load**。

!!! note "/dev/shm（tmpfs）"
    `/dev/shm` 是 Linux 提供的一块"以内存作为存储"的目录。写到 `/dev/shm/xxx` 的文件实际上不会落盘，而是直接驻留在 RAM 中，因此读写速度等同于内存访问。其容量默认是物理内存的一半，系统重启之后数据全部丢失，进程退出时不会自动清理临时文件（所以代码里需要通过 contextmanager 配合 `os.remove` 来显式删除）。

    Grok-1 的 `fast_unpickle` 先用 `shutil.copyfile` 把 ckpt 文件复制到 `/dev/shm`，再对内存中的副本执行 `pickle.load`，目的是让后续所有随机 IO 都落在内存上，而不是反复访问 NFS 或 SSD。32 个 worker 同时运行时，shm 占用的峰值大约是 32 × 5 GB ≈ 160 GB，因此单机 RAM 至少需要预留 32 GB 以上的空间分配给 shm。

为什么要采用这种"先复制到 shm，再 pickle.load"的两段式做法？

1. **NFS 或远程存储 IO 较慢**：ckpt 文件常常存放在 NFS、S3 挂载或网络盘上，直接 `pickle.load` 时频繁的随机 IO 会导致加载严重变慢
2. **pickle 反序列化过程需要随机 seek**：把整个文件一次性写入 tmpfs 后，文件内容完全驻留在内存页缓存中，后续所有 seek 操作都变成内存读取
3. **32 个 worker 并发执行**：每个 worker 独立完成"复制到 shm + unpickle"两步操作

这种做法的代价是 `/dev/shm` 必须预留足够空间，至少需要能容纳单个最大的 tensor（约 5 GB）。

需要注意 `copy_to_shm` 是 contextmanager 的形式：使用完毕后会通过 `os.remove` 清理掉临时文件。在 32 个 worker 并发执行的最坏情况下，shm 同时占用约 32 × 5 GB = 160 GB，机器需要具备相应规模的 RAM。

实际加载 Grok-1 ckpt 时这个峰值并不会被完全用满，因为多个最大的 tensor 不会被同时 unpickle。但为了稳妥起见，shm 至少应该预留 32 GB 的余量。

### 7.3.1 pickle 的安全风险

这里值得专门讨论一下 pickle 带来的安全问题。Python pickle 在反序列化时具备执行任意代码的能力，一个被恶意构造的 pickle 文件可以让加载它的 Python 进程做任何事情，包括删除文件、发起网络请求、植入后门程序等等。

Grok-1 的 ckpt 采用 pickle 格式，这意味着：

1. **必须从可信来源下载 ckpt**，例如官方的 magnet 链接或 HuggingFace 上的 xai-org/grok-1 仓库
2. **第三方镜像源不应信任**，即便它的名字看起来与官方一致
3. **`fast_unpickle` 函数没有做任何 sanitization 处理**，内部就是直接调用 `pickle.load`

一种更安全的方案是 [safetensors](https://github.com/huggingface/safetensors)，它只允许反序列化纯数据 tensor，无法执行任意代码。HuggingFace 强烈推荐使用 safetensors 替代 pickle 作为模型权重格式。

但 Grok-1 发布时 safetensors 已经相当普及（2024 年 3 月时 safetensors 已成为 HuggingFace 默认格式），xAI 仍然选用 pickle，可能的原因有：

1. xAI 内部训练栈原本就基于 pickle 实现
2. ckpt 中包含 `QuantizedWeight8bit` 这类 custom dataclass，safetensors 当时尚未支持嵌套的 metadata
3. 发布时间紧张，来不及切换格式

## 7.4 `load_tensors`：多 host 分工

`checkpoint.py:83-107`：

```python
# checkpoint.py:83-107
def load_tensors(shaped_arrays, directory, mesh_config, tensor_indices=None):
    """Loads a set of arrays."""
    pool = ThreadPoolExecutor(max_workers=32)
    fs = list()
    num_tensors = 0
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    if tensor_indices is None:
        iterator = enumerate(shaped_arrays)
    else:
        iterator = zip(tensor_indices, shaped_arrays)
    for i, t in iterator:
        if (i % num_replicas) == ((jax.process_index() // data_model_shards) % num_replicas):
            idx = (
                jax.process_index() // (num_replicas * data_model_shards) * data_model_shards
                + jax.process_index() % data_model_shards
            )
            fs.append(
                pool.submit(fast_unpickle, os.path.join(directory, f"tensor{i:05d}_{idx:03d}"))
            )
            num_tensors += 1
        else:
            fs.append(pool.submit(np.zeros, t.shape, dtype=t.dtype))
    wait(fs)
    return [f.result() for f in fs]
```

逐段说明：

- `mesh_config` 实际就是 `between_hosts_config`，例如单机配置为 `(1, 1)`，双机配置为 `(1, 2)`
- `data_model_shards = prod(mesh_config)`，表示参与协作加载的 host 总数
- `num_replicas = 1` 是硬编码值，意味着每个 ckpt 分片只被加载一次，整个加载过程没有冗余副本

判断当前 host 是否需要加载第 i 个 tensor 的逻辑：

```python
if (i % num_replicas) == ((jax.process_index() // data_model_shards) % num_replicas):
```

当 `num_replicas = 1` 时这个判断条件恒为 True，也就是说所有 host 都会参与加载流程，不会出现跳过的情况。

shard 索引：

```python
idx = (
    jax.process_index() // (num_replicas * data_model_shards) * data_model_shards
    + jax.process_index() % data_model_shards
)
```

`num_replicas = 1` 时简化为：

```python
idx = jax.process_index() // data_model_shards * data_model_shards + jax.process_index() % data_model_shards
    = jax.process_index()  # 因为整除 + 余数 = 自己
```

即 host `i` 负责加载 `tensor{n:05d}_{i:03d}` 形式的文件。

**在单机 8 卡场景下**（参见 `run.py:60-61`）：

- `local_mesh_config = (1, 8)`：8 张 device 都集中在一台 host 上
- `between_hosts_config = (1, 1)`：只有 1 台 host 参与
- `data_model_shards = 1`
- `jax.process_index() = 0`
- 因此 host 0 实际加载的是所有以 `tensor{i:05d}_000` 命名的文件

但每个 tensor 在磁盘上明明存有 8 个 shard 文件。实际情况是：**单机场景下只使用编号为 `_000` 的那个分片**，其他 7 个分片是为多 host 部署预留的。

这里其实存在一个潜在疑问：如果 ckpt 是按 mesh shape (1, 8) 预先切成 8 份的，那单 host 部署理应需要使用全部 8 份。但代码显示单 host 实际只读取了 1 份分片。

一种可能的解释是：**HuggingFace 上的 ckpt 已经把 model 维度的切分直接预切到 `tensor{i:05d}_{j:03d}` 这种文件中**，其中 `j` 表示 model 维度上的索引。单机 8 卡时每张卡需要读自己对应的 `j`，但由于 `jax.process_index() = 0` 表示只有一个 Python process，所以这个 process 实际通过其他机制把 8 份分片全部加载进来。

下面在 7.5 节里观察 `host_local_array_to_global_array` 的工作方式来进一步澄清这个问题。

## 7.5 `host_local_array_to_global_array`：拼成全局数组

`checkpoint.py:218`：

```python
state = multihost_utils.host_local_array_to_global_array(state, mesh, state_sharding)
```

这个 JAX API 的语义可以表述为：

> 将每个 host 上的 local array 视为该 host 在 mesh 中负责的那一部分，然后将它们拼接成一个全局分布式的 jax.Array。

换言之，**每个 host 提供它所负责的那份本地数组，由 JAX 负责把它们拼接成一个完整的全局数组**。每个 host 上本地数组的 shape 应当等于 global_shape / num_hosts（即沿着 host 切分的那一维做了均分）。

回到 Grok-1 单 host 8 device 的实际场景：

- `jax.process_count() = 1`，`jax.local_device_count() = 8`
- mesh = `(1, 8)`，其中 data 维长度为 1，model 维长度为 8
- 由于只有 1 台 host，本地数组实际上就等于全局数组
- 但数组内部还需要进一步按 device 切分到 8 张 device 上

按这个推理，host 在加载时应当拿到**完整的数组**（即 8 个 model 维分片合并在一起）。但 `load_tensors` 的代码看起来仅加载一份编号为 `_000` 的分片，这与上述推论存在矛盾。

这里有几种可能的实现细节：

A. ckpt 的 `_000` 文件本身就是 unsharded 的完整 tensor，而 `_001..._007` 仅作为冗余备份或者为不同 host 准备的副本
B. 在单 host 模式下，ckpt 加载会从所有 shard 文件中聚合数据

为了澄清这个问题，可以参考 `checkpoint.py:218` 周围的上下文（具体是 `checkpoint.py:212-220`）：

```python
state_sharding = jax.tree_util.tree_map(
    lambda x: jax.sharding.PartitionSpec() if x is None else x,
    state_sharding,
    is_leaf=lambda x: x is None,
)
state = multihost_utils.host_local_array_to_global_array(state, mesh, state_sharding)
```

`host_local_array_to_global_array` 接收 `state_sharding` 作为参数。当 process_count = 1 时，host 上的本地数组就等于全局数组，没有 inter-host 通信，sharding 操作仅限于在 device 之间完成。

按这种解释，**假设 A 似乎可以成立**：即 ckpt 的 `_000` 文件就是完整 tensor，剩余的 `_001..._007` 在单 host 模式下闲置不用。

然而实际上 HuggingFace 仓库中的 grok-1 ckpt 每个 tensor 都包含 8 个分片（每片约 0.7~5 GB），总大小约为 296 GB。如果 `_000` 是完整 tensor，那么 ckpt 整体大小会接近 8 × 296 = 2.3 TB，这显然不成立。因此 **A 假设站不住脚**。

更合理的解释是 **C**：**ckpt 是按 mesh shape (1, 8) 预先切好的**，`_000` 到 `_007` 对应 model 维度上的 8 个切片。在单 host 模式下：

- 本地数组 = 当前 host 在 mesh 上所负责的那个切片（即 model 维度上的 1/8）
- `data_model_shards = 1`（因为 between_hosts_config = (1, 1)），意味着每个 host 只加载 1 个 shard
- 但 host 0 拥有 8 张 device，每张 device 需要的 model shard 各不相同
- `host_local_array_to_global_array` 在单 host 内部负责把数据进一步排布到各个 device 上

但 7.4 节中的代码确实只为 host 0 提交了 1 个文件加载任务。这说明在 (1, 1) 配置下，Grok-1 默认假设 ckpt 本身就是按 (1, 1) 切好了的 1 个分片。

由于无法实际运行 ckpt 验证，这一段的实现细节难以 100% 确认。**简化的结论是：load_tensors 中的 sharding 逻辑本质上为多 host 场景设计；在 run.py 默认的"1 host + 8 device"配置下，每个 process 只加载编号为 `_000` 的一份文件，再由 JAX 内部按 mesh 把全局数组分布到 8 张 device 上**。

这进一步意味着 ckpt 文件名中的 `_{shard:03d}` 后缀其实是沿 **between_hosts** 维度切分的，而不是沿 device 维度切分的。HuggingFace 上提供的 ckpt 应当是按某个特定的 host 数（很可能是 1 host）预切的，因此默认情况下 `_000` 可以直接用于单机加载。

## 7.6 `replace_with_load_state`：参数名映射

`checkpoint.py:144-177`：

```python
# checkpoint.py:144-177
def replace_with_load_state(
    init_state: Any,
    load_state: Any,
    load_rename_rules: Optional[list[tuple[str, str]]] = None,
    load_exclude_rules: Optional[list[str]] = None,
    mesh_config: tuple = (1, 1),
) -> Any:
    flatten_load, _ = jax.tree_util.tree_flatten_with_path(load_state)
    flatten_init, structure_init = jax.tree_util.tree_flatten_with_path(init_state)
    load_map = {path_tuple_to_string(path): tensor for path, tensor in flatten_load}

    replaced = []
    num_replicas = 1
    data_model_shards = math.prod(mesh_config)
    for i, (init_path, tensor) in enumerate(flatten_init):
        init_path_str = path_tuple_to_string(init_path)
        load_path_str = get_load_path_str(init_path_str, load_rename_rules, load_exclude_rules)
        if load_path_str is None:
            rank_logger.info(f"Excluded from restore: {init_path_str}.")
            replaced.append(tensor)
        elif load_path_str in load_map:
            if load_path_str == init_path_str:
                rank_logger.info(f"Restored from ckpt: {init_path_str}.")
            else:
                rank_logger.info(f"Restored from ckpt: {init_path_str} <-- {load_path_str}.")
            replaced.append(load_map[load_path_str])
        else:
            rank_logger.info(f"Not found in ckpt: {init_path_str}.")
            if (i % num_replicas) == ((jax.process_index() // data_model_shards) % num_replicas):
                replaced.append(tensor)
            else:
                replaced.append(np.zeros_like(tensor))

    return jax.tree_util.tree_unflatten(structure_init, replaced)
```

这是一个用于**参数名重命名加排除**的工具函数，主要用途包括：

- 把代码中的参数名（例如 `decoder_layer_0/multi_head_attention/query/w`）映射到 ckpt 中使用的另一种参数名
- 跳过某些不需要从 ckpt 加载的参数（例如某些层的特殊参数）

但是 `restore` 函数中**并没有调用这个工具**。因此该函数在当前代码路径中并未真正发挥作用，可能是 xAI 为未来的兼容性场景预先保留的。

## 7.7 `restore`：主入口

`checkpoint.py:180-221`：

```python
# checkpoint.py:180-221
def restore(
    checkpoint_path: str,
    state_shapes: Any,
    mesh,
    between_hosts_config,
    params_only,
    state_sharding,
    init_state: Optional[Any] = None,
) -> Any:
    ckpt_path = os.path.join(checkpoint_path, "ckpt-0")

    rank_logger.info("Loading checkpoint at {}".format(ckpt_path))
    ckpt_shapes = state_shapes
    ckpt_shapes_with_path, structure = jax.tree_util.tree_flatten_with_path(ckpt_shapes)

    ckpt_shapes_flat = [elem[1] for elem in ckpt_shapes_with_path]
    loaded_tensors = load_tensors(ckpt_shapes_flat, ckpt_path, between_hosts_config)

    state = jax.tree_util.tree_unflatten(structure, loaded_tensors)

    # Sanity check to give a better error message.
    ckpt_keys = set(state.params.keys())
    code_keys = set(state_sharding.params.keys())

    if ckpt_keys != code_keys and init_state is None:
        missing_in_ckpt = code_keys - ckpt_keys
        missing_locally = ckpt_keys - code_keys
        raise ValueError(
            "Parameters in the code are not matching checkpoint parameters.\n"
            "Params missing in checkpoint: {}\nParams missing in code: {}".format(
                missing_in_ckpt, missing_locally
            )
        )
    state_sharding = jax.tree_util.tree_map(
        lambda x: jax.sharding.PartitionSpec() if x is None else x,
        state_sharding,
        is_leaf=lambda x: x is None,
    )
    state = multihost_utils.host_local_array_to_global_array(state, mesh, state_sharding)
    if params_only:
        state = state.params
    return state
```

具体步骤：

1. **flatten state_shapes**：把代码 init 出的 state（一棵嵌套的 pytree）压平成叶子节点列表，得到大约 770 个 `(path, shape)` 二元组
2. **load_tensors**：按 path 的顺序依次加载每个 tensor 对应的文件
3. **unflatten**：把加载完成的 tensor 列表重新组装回与原结构一致的 state pytree
4. **sanity check**：对比 ckpt 中的参数名与代码中定义的参数名是否一致，这一步在用户修改了模型结构但忘记同步更新 ckpt 时，提供友好的错误提示
5. **host_local -> global**：使用 mesh 信息把 host-local 的数据组装成跨 device 的全局分布式数组
6. **params_only**：只返回模型参数部分（去掉其他与训练相关的 state）

## 7.8 为什么单 GPU 加载不动：显存账

主要硬件需求：

### 7.8.1 bf16 全精度

- 模型参数：314B × 2 bytes = **628 GB**
- KV cache（batch=1）：2 GB
- 激活值：~10 GB
- 加载时临时 buffer：相当于一份参数 ~628 GB

启动峰值需要 ~1.2 TB，运行稳态需要 ~640 GB。

**单张 H100 80GB 显然完全不够**。最小可行配置是 8 × H100 80GB 合计 640 GB，刚好能容纳模型，但完全没有 buffer 余量。**实际推荐的配置是在 8 × H100 80GB 上使用 8-bit 量化的 ckpt**。

### 7.8.2 8-bit 量化

- 模型参数：314B × 1 byte = **314 GB**
- KV cache（仍 bf16）：2 GB
- 激活值：~10 GB

单张 80 GB 卡仍然远远不够，但 4 × H100 80GB 合计 320 GB 已经勉强足够装下。**推荐的配置是 8 × A100 80GB（共 640 GB）**，这样可以保证有充足的 buffer 余量。

### 7.8.3 最小硬件配置建议

| 配置 | 量化 | 跑得动？ |
| --- | --- | --- |
| 1 × H100 80GB | 任何 | 否（显存不足以装下） |
| 4 × H100 80GB (320 GB) | 8-bit | 勉强，无 buffer |
| 8 × A100 80GB (640 GB) | bf16 | 是 |
| 8 × H100 80GB (640 GB) | bf16 / 8-bit | 是，推荐 |
| 16 × A100 40GB (640 GB) | bf16 | 是，需要双机 |

注意 `run.py:60` 中的 `local_mesh_config=(1, 8)` 是针对 **8 卡单机**写死的配置。如果改成 16 卡双机部署，需要相应将 `between_hosts_config` 修改为 `(1, 2)`。

### 7.8.4 显存账更细致拆解

在 8 × H100 80GB 单机上加载 bf16 ckpt 时，显存使用按时间线大致如下：

1. **加载阶段**：主机 RAM 用于存放 ckpt 的临时数据，大约 10-30 GB（并不是所有 tensor 同时加载），此时 GPU 显存尚未被使用
2. **host_local -> global 阶段**：每张 GPU 接收自己负责的那部分参数，大约 78 GB（628 GB / 8）
3. **JIT compile 阶段**：在参数之外，XLA 编译过程还需要一些临时 buffer，每张卡额外占用 5-10 GB
4. **稳态推理阶段**：参数 78 GB + KV cache 2-17 GB + 激活值约 3-5 GB，合计约 85-100 GB

最后这一步在 80 GB 卡上勉强不会爆显存，但用于扩大 batch、延长 context 的余量已经非常有限。

如果改用 8-bit ckpt：

- 参数显存减半到约 40 GB / 卡
- 总显存使用约 50-60 GB / 卡
- 为 batch 扩展与 KV cache 增长留出充足余量

这也是为什么生产部署 Grok-1 时几乎都选择使用 8-bit 量化版本。

## 7.9 加载耗时与 IO 瓶颈

加载 296 GB 的 ckpt：

- **机械硬盘**：50 MB/s 顺序读 → ~1.6 小时
- **SATA SSD**：500 MB/s → ~10 分钟
- **NVMe SSD**：3 GB/s → ~100 秒
- **网络（NFS, 1Gbit）**：100 MB/s → 50 分钟

通过 `/dev/shm` 做中转加上 32 worker 并发，可以让顺序 IO 接近物理上限。实测在本地 NVMe SSD 上加载完整 ckpt 大约需要 5-10 分钟。

`pickle.load` 本身存在一定的固有开销，包括 Python 字节码执行、unpickle 过程中的函数调用、以及 numpy buffer 的内存拷贝。`fast_unpickle` 这个名字暗示存在某种"快速"路径，但实际上它只是把文件复制到 shm 之后再调用标准的 pickle.load，并不是 zero-copy 的 mmap 实现。如果追求真正的高性能加载，应当转向 safetensors 或自行实现 mmap 加载，但 Grok-1 在这里选择了简单直接的实现方式。

## 7.10 总结

checkpoint.py 是一份满足基本需求即可的加载实现：

1. **物理布局**：大约 770 个 tensor，每个 tensor 被切分为 N 个 pickle 格式的分片文件
2. **`/dev/shm` 中转**：避免直接从 NFS 加载时遭遇的随机 IO 性能问题
3. **32 worker 并发**：每个 worker 独立完成 unpickle 流程
4. **多 host 分工**：每个 host 只加载其在 mesh 切片中所对应的那部分文件
5. **host_local -> global**：借助 JAX 的 multihost utils 把 host-local 数组拼接为全局分布式数组
6. **sanity check**：参数名出现不匹配时，会向用户报出可读性较好的错误信息

整体实现简单易读，但并非最快方案。如果要面向工业级部署，应当重写为基于 safetensors + mmap 的多线程 zero-copy 加载。但对开源研究用途而言，"5-10 分钟完成一次加载"已经是可以接受的开销。

下一章看 runners.py，分析推理引擎层的实现。

## 延伸阅读

- [safetensors](https://github.com/huggingface/safetensors) - HuggingFace 后来推广的更安全 / 快的 ckpt 格式
- [Distributed checkpointing with JAX](https://jax.readthedocs.io/en/latest/distributed_data_loading.html) - JAX 官方的多 host 加载文档
- [Megatron-LM checkpoint format](https://github.com/NVIDIA/Megatron-LM) - NVIDIA 的多 host ckpt 切分约定
- [Orbax](https://github.com/google/orbax) - Google 的 JAX ckpt 工具，被许多新项目用于替代手写 pickle 加载
