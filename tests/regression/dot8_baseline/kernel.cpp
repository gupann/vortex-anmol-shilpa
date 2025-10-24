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

    for (int k = 0; k < size; ++k) {
        int8_t a = A[row * size + k];
        int8_t b = B[k * size + col];
        sum += static_cast<int32_t>(a) * static_cast<int32_t>(b);
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
