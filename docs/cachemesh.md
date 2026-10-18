# Bloom 分层缓存

Bloom 的 IFB 路径提供两种互补的分层缓存接口：`CacheMesh` 负责 Candle 原生
KV block 的模型布局适配，`LayeredBlockCache` 提供与模型无关的字节级 block
管理。两者都在 `bloomai-engine` 内，不依赖 HTTP 服务，因此可以嵌入桌面端、
批处理程序或其他运行时。

## 运行时 CacheMesh

启动 IFB 和主机内存 L2：

```bash
cargo run --release -p bloomai-server --bin bloom_server -- \
  --model /path/to/model \
  --enable-ifb \
  --enable-cachemesh \
  --cachemesh-l2-capacity-bytes 2147483648
```

增加可跨进程、可跨重启复用的文件 L3：

```bash
cargo run --release -p bloomai-server --bin bloom_server -- \
  --model /path/to/model \
  --enable-ifb \
  --enable-cachemesh \
  --enable-cachemesh-l3 \
  --cachemesh-l3-path ~/.cache/bloom/kv \
  --cachemesh-write-through-l3
```

L2 使用字节容量和 LRU 淘汰，L3 文件写入采用临时文件加原子替换。L3 文件
会校验路径类型、请求 key、JSON 形状和最大读取大小；坏文件会返回错误并计入
失败指标。运行时状态可以从 `/v1/observability` 或 `/metrics` 查看，KV 专项
统计在 `/v1/kv-cache-stats`。

这些 server flags 作用于 IFB 的 Candle 原生 KV 路径；其他模型后端不会因为
打开 CacheMesh 而自动获得 KV 分层。需要在其他执行器中复用分层能力时，使用
下面的独立 `LayeredBlockCache` API。

`CacheMeshKey` 的 namespace 应包含模型版本或权重 revision。生产环境不要让
不同权重共用同一个 namespace，否则旧的 prefix KV 可能被重新使用。服务端默认
使用加载模型的标识；模型仓库更新时请使用新的模型标识或清理 L3 目录。

## 独立字节级分层缓存

`LayeredBlockCache` 不要求 Candle，也不暴露物理 GPU block。它提供：

- L1/L2 有界内存 LRU，按字节而不是条目数计费；
- L1 命中、L2 命中和 L3 命中自动晋升；淘汰按 L1 → L2 → L3 顺序降级；
- `pin`/`unpin` 保护正在使用的 block；
- 可插拔 `LayeredCacheBackend`，以及带版本、长度、SHA-256 校验和、原子写入
  的 `FileSystemLayeredCache`；
- 每层的命中率、晋升、淘汰、容量和后端错误统计。

仓库内的端到端示例可以直接运行：

```bash
cargo run -p bloomai-engine --example layered_cache
```

最小示例：

```rust
use std::sync::Arc;
use bloomai_engine::{
    FileSystemLayeredCache, LayeredBlockCache, LayeredCacheConfig,
};

let l3 = Arc::new(FileSystemLayeredCache::new("/var/cache/bloom/blocks")?);
let cache = LayeredBlockCache::with_backend(
    LayeredCacheConfig {
        l1_capacity_bytes: 64 * 1024 * 1024,
        l2_capacity_bytes: 512 * 1024 * 1024,
        max_entry_bytes: 256 * 1024 * 1024,
        write_through: false,
    },
    Some(l3),
)?;

cache.put("model-revision/layer-0/block-42", encoded_block)?;
let block = cache.get("model-revision/layer-0/block-42")?;
```

`put`、`get`、`remove`、`pin` 和 `unpin` 可以从多个线程并发调用。后端 I/O
发生在释放内存锁之后；后端故障会返回错误并增加 `backend_errors`，不会伪造
缓存命中。需要直接控制 L3 的部署可以实现 `LayeredCacheBackend`，例如接入
共享对象存储或带租约的远端 block 服务。

## 容量与故障策略

`max_entry_bytes` 是单个值的硬上限，应该小于服务进程允许的最大请求或 KV
block 大小。L1/L2 设为零表示关闭该内存层；如果两个内存层都为零，必须提供
L3 后端。启用 `write_through` 时，L3 写入成功后 `put` 才会返回；普通模式将
L3 用作淘汰后的恢复层。应用应把后端错误记录到自己的重试或降级策略中，并在
部署时为缓存目录设置独立配额和访问权限。
