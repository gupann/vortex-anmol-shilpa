#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<int8_t*>(arg->A_addr);
	auto B = reinterpret_cast<int8_t*>(arg->B_addr);
	auto C = reinterpret_cast<int32_t*>(arg->C_addr);
    auto size = static_cast<int>(arg->size);

    int col = blockIdx.x;
    int row = blockIdx.y;

    int32_t sum(0);

    for (int k = 0; k + 3 < size; k += 4) {
        // Pack 4 int8_t elements from A and B into 32-bit integers
        uint32_t packedA;
        std::memcpy(&packedA, &A[row * size + k], sizeof(uint32_t));

        uint32_t packedB =
            (uint32_t)(uint8_t)B[(k+0)*size + col]        |
            (uint32_t)(uint8_t)B[(k+1)*size + col] << 8   |
            (uint32_t)(uint8_t)B[(k+2)*size + col] << 16  |
            (uint32_t)(uint8_t)B[(k+3)*size + col] << 24;

        sum += vx_dot8(packedA, packedB);
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
