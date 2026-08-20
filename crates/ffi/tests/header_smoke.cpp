#include "../bloom.h"

int main() {
    BloomSlice input{};
    BloomOwnedBuffer output{};
    BloomStatus status = BLOOM_STATUS_OK;
    return input.len == 0 && output.len == 0 && status == BLOOM_STATUS_OK &&
                   BLOOM_ABI_VERSION >= 2u
               ? 0
               : 1;
}
