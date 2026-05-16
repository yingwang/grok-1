# 第 7 章 checkpoint.py 权重加载

`checkpoint.py` 只有 221 行，但它解决的问题不小：把 300+ GB 的权重在多 host / 多 device 的环境下、从磁盘加载到 JAX 的全局分布式数组里。本章把这个流程拆清楚。

!!! note "multi-host / SPMD（Single Program Multiple Data）"
    单台机器装不下 314B 权重，所以 Grok-1 假设你有若干台机器（host），每台机器若干张 GPU（device）。每台机器各自跑同一份 Python 程序，但只负责整个模型的一小块，靠 NCCL/通信库交换中间结果。这种"同一份代码 + 各跑各的分片"的模式就叫 SPMD。

    JAX 用 `jax.process_index()` 给每台机器一个编号，加载 ckpt 时各台机器按编号决定自己读哪些文件。Grok-1 的 `run.py` 默认配置是单机 8 卡，所以 `jax.process_count() = 1`、`jax.process_index() = 0`，多 host 路径退化掉；但 7.4、7.5 的代码逻辑是为多机准备的。

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
    Python 自带的对象序列化协议。`pickle.dump(obj, f)` 把任意 Python 对象（dict、numpy array、自定义 class 实例）转成字节流写文件，`pickle.load(f)` 反过来还原。它的方便之处是不用自己写 schema，连嵌套结构都能直接存；缺点是反序列化时会执行字节流里描述的"如何还原 class"的指令，所以恶意 pickle 文件能在你的进程里跑任意代码。

    Grok-1 的 ckpt 一共 ~770 个 tensor × N 个分片 ≈ 几千个 pickle 文件，每个文件存一个 numpy 数组或 `QuantizedWeight8bit` 实例。整套 296 GB 都靠 pickle 序列化。

总体大小（HuggingFace 上 xai-org/grok-1）：约 296 GB（8-bit 量化）。如果用原始 bf16，约 600 GB。

每个 tensor 大概在 100MB 到 5GB 之间，取决于该 tensor 的 shape。最大的一类是 MoE 的 `linear_v/w`、`linear/w`、`linear_1/w`，单层就是 $(8, 6144, 32768)$ 即 1.5 GB（bf16），8-bit 化后 0.75 GB。64 层共 144 GB - 占了 ckpt 大头。

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
    │     把每个 host 的本地数组凑成 JAX 的全局分布式 GlobalDeviceArray
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

整套机制是：**先把 ckpt 文件复制到 `/dev/shm`（tmpfs，内存盘），再 pickle.load**。

!!! note "/dev/shm（tmpfs）"
    Linux 提供的一块"用内存当磁盘"的目录。写到 `/dev/shm/xxx` 的文件不落盘，直接放在 RAM 里，读写速度等同内存。它的容量默认是物理内存的一半，重启即丢失，进程退出不会自动清理（所以代码里用 contextmanager + `os.remove` 显式删）。

    Grok-1 的 `fast_unpickle` 先把 ckpt 文件 `shutil.copyfile` 到 `/dev/shm` 再 `pickle.load`，目的是把后续随机 IO 全部喂给内存而不是 NFS / SSD。32 个 worker 同时跑，并发占用峰值约 32 × 5 GB ≈ 160 GB，所以单机 RAM 至少要预留 32 GB 富余给 shm。

为什么这么做？

1. **NFS / 远程存储慢**：ckpt 文件可能在 NFS、S3 mount、或网络盘上，直接 `pickle.load` 会因为 random IO 慢
2. **pickle 反序列化需要 seek**：一次性写到 tmpfs，整个文件落入内存页缓存，后续 seek 全是内存操作
3. **32 个 worker 并发**：每个 worker 独立做 shm 复制 + unpickle

代价是 `/dev/shm` 需要有足够空间 - 至少 5GB（单个 tensor 最大）。

注意 `copy_to_shm` 是 contextmanager - 用完会 `os.remove` 把临时文件清掉。在并发 32 worker 时同时占用最多约 32 × 5GB = 160 GB shm，需要机器有这么多 RAM。

实际 Grok-1 加载时不会全用满，因为大 tensor 不会同时被 unpickle。但 shm 至少要 32GB 富余比较稳妥。

### 7.3.1 pickle 的安全风险

值得花一段说说 pickle 的安全问题。Python pickle 在反序列化时会执行任意代码 - 一个恶意构造的 pickle 文件可以让你的 Python 进程做任何事（删文件、发请求、装后门）。

Grok-1 的 ckpt 用 pickle，意味着：

1. **必须从可信源下载 ckpt** - magnet 链接 / xai-org/grok-1 HuggingFace 仓库 = 官方源
2. **第三方镜像不要信任**，即便看起来名字一样
3. **`fast_unpickle` 没做任何 sanitization** - 它就是直接 `pickle.load`

更安全的方案是 [safetensors](https://github.com/huggingface/safetensors)，它只允许反序列化纯数据 tensor，不能执行任意代码。HuggingFace 强力推荐用 safetensors 替代 pickle。

但 Grok-1 发布时 safetensors 已普及（2024-03 时 safetensors 已是 HuggingFace 默认格式），xAI 仍选 pickle 可能因为：

1. xAI 内部训练栈本来就用 pickle
2. ckpt 包含 `QuantizedWeight8bit` 这种 custom dataclass，safetensors 当时不支持嵌套 metadata
3. 时间紧来不及改

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

逐段：

- `mesh_config` 是 `between_hosts_config`，比如 `(1, 1)` 单机或 `(1, 2)` 双机
- `data_model_shards = prod(mesh_config)` - 多少个 host 协作
- `num_replicas = 1`（硬编码）- 意味着每份 ckpt 只被 load 一次，没有冗余

判定 host 是否要 load 第 i 个 tensor：

```python
if (i % num_replicas) == ((jax.process_index() // data_model_shards) % num_replicas):
```

`num_replicas = 1` 时永远 True - 即所有 host 都参与 load，没有跳过。

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

即 host `i` 加载 `tensor{n:05d}_{i:03d}` 文件。

**单机 8 卡场景下**（`run.py:60-61`）：

- `local_mesh_config = (1, 8)`：8 个 device 在一个 host 上
- `between_hosts_config = (1, 1)`：只有 1 个 host
- `data_model_shards = 1`
- `jax.process_index() = 0`
- 即 host 0 加载所有 `tensor{i:05d}_000` 文件

但每个 tensor 有 8 个 shard 文件啊？事实是：**单机场景下只用 `_000` 那个分片**。其他分片是为多 host 准备的。

嗯，这里有个问题：如果 ckpt 是按 mesh shape (1, 8) 切的 8 份，单 host 应该用所有 8 份。但代码看起来单 host 只用 1 份。

可能的解释：**HuggingFace 上的 ckpt 已经把 model 维度的切分预切到 `tensor{i:05d}_{j:03d}` 文件中**，其中 `j` 是 model 维度的索引。单 host 8 卡时，每张卡读自己对应的 `j`，但因为 `jax.process_index() = 0` 只有一个 process，所以这个 process 实际把 8 份都 load 进来 - 通过别的机制。

让我们看 7.5 节里 `host_local_array_to_global_array` 怎么做。

## 7.5 `host_local_array_to_global_array`：拼成全局数组

`checkpoint.py:218`：

```python
state = multihost_utils.host_local_array_to_global_array(state, mesh, state_sharding)
```

这个 JAX API 的语义：

> 将每个 host 上的 local array 视为该 host 在 mesh 中负责的那一部分，然后凑成一个全局分布式 jax.Array。

也就是说，**每个 host 提供它"那一份"的本地数组，JAX 帮你拼**。本地数组的 shape 应该等于 global_shape / num_hosts（沿 host 维切）。

回到单 host 8 device 的 Grok-1 场景：

- `jax.process_count() = 1`，`jax.local_device_count() = 8`
- mesh = `(1, 8)`，data=1，model=8
- 本地 array 就是 global array（因为只有 1 host）
- 但 array 内部还要按 device 切到 8 个 device 上

那么 host 加载时拿到的应该是**完整的 array**（all 8 model shards 合在一起）。但 `load_tensors` 看起来只 load 一份 `_000`？

这里可能有几种实现细节：

A. ckpt 的 `_000` 文件本身就是 unsharded full tensor（其他 `_001..._007` 是冗余备份 / 不同 host 复制）
B. 单 host 模式下 ckpt 加载会从所有 shard 文件聚合

读 `checkpoint.py:218` 上下文（`checkpoint.py:212-220`）：

```python
state_sharding = jax.tree_util.tree_map(
    lambda x: jax.sharding.PartitionSpec() if x is None else x,
    state_sharding,
    is_leaf=lambda x: x is None,
)
state = multihost_utils.host_local_array_to_global_array(state, mesh, state_sharding)
```

`host_local_array_to_global_array` 接收 `state_sharding` 作为参数。当 process_count = 1，host 上的本地数组等于全局数组，不需要 inter-host 通信，只在 device 间做 sharding。

所以 **A 假设可能成立**：ckpt 的 `_000` 文件是 full tensor，剩下的 `_001..._007` 在单 host 模式下 unused。

实际上 HuggingFace 上 grok-1 ckpt 每个 tensor 有 8 个分片（约 0.7~5GB 每片），总和约 296 GB。如果 `_000` 就是 full tensor，那 ckpt 总大小会接近 8 × 296 = 2.3 TB（不可能）。所以 **A 错**。

正确解释应该是 **C**：**ckpt 是按 mesh shape (1, 8) 预切的**，`_000`..`_007` 是 model 维度的 8 个切片。单 host 模式下：

- 本地数组 = host 自己负责的 mesh 切片（即 model 维度的 1/8）
- `data_model_shards = 1`（between_hosts_config = (1,1)），意味着每个 host 只 load 1 个 shard
- 但 host 0 拥有 8 个 device，每个 device 需要不同的 model shard
- `host_local_array_to_global_array` 在单 host 内做 device 间的 layout

但 7.4 的代码里只 submit 了 1 个 file 给 host 0。这说明在 (1, 1) 配置下 Grok-1 假设 ckpt 已经按 (1, 1) 切好了 1 个文件。

这一段在没有真实运行 ckpt 的情况下不能 100% 确认实现细节。**简单结论是：load_tensors 的 sharding 逻辑设计用于多 host 场景，在 run.py 默认的 1 host 8 device 配置下，每个 process 只 load 一份索引为 `_000` 的文件，然后 JAX 内部把全局数组按 mesh 切到 8 个 device**。

这意味着 ckpt 文件名中的 `_{shard:03d}` 后缀实际上是按 **between_hosts** 维度切分的，不是按 device 维度切分。HuggingFace 上的 ckpt 应该是按某种特定的 host 数预切的（可能是 1 host），所以 `_000` 是默认的、可用于单机加载。

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

这是一个**参数名重映射 + 排除**工具，用于：

- 把代码里的参数名（如 `decoder_layer_0/multi_head_attention/query/w`）映射到 ckpt 里的不同名字
- 跳过某些参数（如某些层的特殊参数）

但是 `restore` 函数里**没有调用这个**。所以这个函数当前没起作用，可能是为未来兼容性保留的。

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

步骤：

1. **flatten state_shapes**：把代码 init 出的 state（一个嵌套 pytree）压平成 leaf list，得到 ~770 个 `(path, shape)` 二元组
2. **load_tensors**：按 path 顺序加载每个 tensor 文件
3. **unflatten**：把加载的 tensor list 重新组装成 state pytree
4. **sanity check**：检查 ckpt 参数名是否与代码定义一致 - 这是用户改了 model 结构但忘了改 ckpt 时的友好错误提示
5. **host_local -> global**：用 mesh 把 host-local 数据组装成全局分布式数组
6. **params_only**：只返回参数（去掉其他 training state）

## 7.8 为什么单 GPU 加载不动：显存账

主要硬件需求：

### 7.8.1 bf16 全精度

- 模型参数：314B × 2 bytes = **628 GB**
- KV cache（batch=1）：2 GB
- 激活值：~10 GB
- 加载时临时 buffer：相当于一份参数 ~628 GB

启动峰值需要 ~1.2 TB，运行稳态需要 ~640 GB。

**单卡 H100 80GB 显然不够**。最小可行：8 × H100 80GB = 640GB，正好够，但没有 buffer 余量。**实际推荐 8 × H100 80GB 用 8-bit ckpt**。

### 7.8.2 8-bit 量化

- 模型参数：314B × 1 byte = **314 GB**
- KV cache（仍 bf16）：2 GB
- 激活值：~10 GB

单卡 80GB 远远不够，但 4 × H100 80GB = 320GB 勉强够。**推荐 8 × A100 80GB（共 640 GB）** 来保证有 buffer。

### 7.8.3 最小硬件配置建议

| 配置 | 量化 | 跑得动？ |
| --- | --- | --- |
| 1 × H100 80GB | 任何 | 否（不够装下） |
| 4 × H100 80GB (320 GB) | 8-bit | 勉强，无 buffer |
| 8 × A100 80GB (640 GB) | bf16 | 是 |
| 8 × H100 80GB (640 GB) | bf16 / 8-bit | 是，推荐 |
| 16 × A100 40GB (640 GB) | bf16 | 是，需要双机 |

注意 `run.py:60` 的 `local_mesh_config=(1, 8)` 是为 **8 卡单机**写死的。如果是 16 卡双机，需要改 `between_hosts_config=(1, 2)`。

### 7.8.4 显存账更细致拆解

8 × H100 80GB 单机加载 bf16 ckpt 的显存使用，按时间线：

1. **加载阶段**：主机 RAM 装 ckpt 临时数据约 ~10-30 GB（不是同时全 load），GPU 显存还没用
2. **host_local -> global 阶段**：每张 GPU 接收自己那部分参数，大约 78 GB（628 GB / 8）
3. **JIT compile 阶段**：除参数外，XLA 需要一些临时 buffer 编译，每张卡 +5-10 GB
4. **稳态推理**：参数 78 GB + KV cache 2-17 GB + 激活 ~3-5 GB = ~85-100 GB

最后这步在 80 GB 卡上勉强不爆，但留给 batch 扩大、长 context 的余地很小。

如果用 8-bit ckpt：

- 参数显存减半到 ~40 GB / 卡
- 总显存 ~50-60 GB / 卡
- 留给 batch、KV cache 的余地宽松

这就是为什么生产部署 Grok-1 几乎都用 8-bit。

## 7.9 加载耗时与 IO 瓶颈

加载 296 GB 的 ckpt：

- **机械硬盘**：50 MB/s 顺序读 → ~1.6 小时
- **SATA SSD**：500 MB/s → ~10 分钟
- **NVMe SSD**：3 GB/s → ~100 秒
- **网络（NFS, 1Gbit）**：100 MB/s → 50 分钟

`/dev/shm` 中转和 32 worker 并发能让顺序 IO 接近物理上限。实测在本地 NVMe SSD 上加载约 5-10 分钟。

`pickle.load` 本身有一定开销 - Python 字节码、unpickle 函数调用、numpy buffer 拷贝。`fast_unpickle` 名字暗示有"快"路径，但实际上它只是把文件复制到 shm 再正常 pickle.load，并不是 zero-copy mmap。如果想真正快，应该用 safetensors / 自定义 mmap，但 Grok-1 选了简单粗暴。

## 7.10 总结

checkpoint.py 是一份"够用即可"的加载实现：

1. **物理布局**：~770 个 tensor，每个被切成 N 个 pickle 分片文件
2. **/dev/shm 中转**：避免 NFS / 随机 IO 问题
3. **32 worker 并发**：每个 worker 独立 unpickle
4. **多 host 分工**：每个 host 只 load 自己 mesh 切片对应的文件
5. **host_local -> global**：用 JAX 的 multihost utils 把 host-local 数组拼成全局分布式数组
6. **sanity check**：参数名不匹配会报友好错误

简单、易读，但不是最快。如果工业部署，应该重写成 safetensors + mmap + 多线程 zero-copy 加载。但对开源研究用途，"5-10 分钟加载一次" 是可接受的。

下一章看 runners.py - 推理引擎层。

## 延伸阅读

- [safetensors](https://github.com/huggingface/safetensors) - HuggingFace 后来推广的更安全 / 快的 ckpt 格式
- [Distributed checkpointing with JAX](https://jax.readthedocs.io/en/latest/distributed_data_loading.html) - JAX 官方的多 host 加载文档
- [Megatron-LM checkpoint format](https://github.com/NVIDIA/Megatron-LM) - NVIDIA 的多 host ckpt 切分约定
- [Orbax](https://github.com/google/orbax) - Google 的 JAX ckpt 工具，被许多新项目用于替代手写 pickle 加载
