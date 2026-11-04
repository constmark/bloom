;; ABI-v1 inline block processor. Replaces the ASCII z byte with Z.
;; Envelope keys contain no z; Unicode bytes and tensor/token arrays survive.
;; A fresh invocation must always see call_count=0, proving isolation.
(module
  (memory (export "memory") 1 128)
  (global $call_count (mut i32) (i32.const 0))
  (func (export "bloom_plugin_abi_version") (result i32) i32.const 1)
  (func (export "bloom_alloc") (param $len i32) (result i32)
    (local $pages i32)
    local.get $len i32.const 65535 i32.add i32.const 16 i32.shr_u
    local.tee $pages memory.size i32.gt_u
    if local.get $pages memory.size i32.sub memory.grow drop end
    i32.const 0)
  (func (export "bloom_process") (param $ptr i32) (param $len i32) (result i64)
    (local $cursor i32) (local $end i32)
    global.get $call_count if unreachable end
    i32.const 1 global.set $call_count
    local.get $ptr local.set $cursor
    local.get $ptr local.get $len i32.add local.set $end
    block $done
      loop $next
        local.get $cursor local.get $end i32.ge_u br_if $done
        local.get $cursor i32.load8_u i32.const 122 i32.eq
        if local.get $cursor i32.const 90 i32.store8 end
        local.get $cursor i32.const 1 i32.add local.set $cursor
        br $next
      end
    end
    local.get $len i64.extend_i32_u i64.const 32 i64.shl
    local.get $ptr i64.extend_i32_u i64.or))
