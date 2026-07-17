#include <cstdio>
#include <cuda_runtime.h>
int main() {
    fprintf(stderr, "step 1: start\n");
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "step 2: device=%s err=%d\n", prop.name, err);
    fflush(stderr);
    return 0;
}
