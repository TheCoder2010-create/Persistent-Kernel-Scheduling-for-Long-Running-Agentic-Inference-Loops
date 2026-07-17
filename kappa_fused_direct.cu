// kappa_fused_direct.cu
//
// Direct measurement of κ_fused: the overhead of relaunching the fused
// transformer-step kernel once per step, WITHOUT subtraction of two
// large noisy CUDA-event measurements.
//
// Approach:
//   1. Launch transformer_step_kernel N=1000 times in a tight loop,
//      reusing the same input buffer (no memcpy between launches).
//   2. Record per-launch CUDA event pairs → median total time (compute + κ).
//   3. The kernel also records clock64 timestamps → compute time in cycles.
//   4. κ_fused = median(per_launch_total) - clock64_compute_cycles / clock_rate.
//
// Build:
//   nvcc -O3 -arch=sm_89 kappa_fused_direct.cu -o kappa_fused_direct
//
// Run:
//   kappa_fused_direct --json-out kappa_fused_direct.json
//
// For N=10 independent runs, use the companion script or batch file.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cassert>
#include <string>

// ---------------------------------------------------------------------------
// Configuration (matches persistent_kernel.cu)
// ---------------------------------------------------------------------------
#define DIM          256
#define N_HEADS      4
#define HEAD_DIM     (DIM / N_HEADS)
#define MLP_RATIO    4
#define FFN_DIM      (DIM * MLP_RATIO)
#define MAX_SEQ_LEN  512
#define N_LAYERS     2
#define EPS          1e-5f

// ---------------------------------------------------------------------------
// Helper macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), #call); \
        exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Weight layout helpers (copied from persistent_kernel.cu)
// ---------------------------------------------------------------------------
#define LAYER_FLOATS (2*DIM + DIM*3*DIM + 3*DIM + DIM*DIM + DIM + 2*DIM + DIM*FFN_DIM + FFN_DIM + FFN_DIM*DIM + DIM)

__host__ __device__ inline int layer_offset(int layer) { return layer * LAYER_FLOATS; }

__host__ __device__ inline int off_ln1_w(int layer)  { return layer_offset(layer) + 0; }
__host__ __device__ inline int off_ln1_b(int layer)  { return off_ln1_w(layer) + DIM; }
__host__ __device__ inline int off_qkv_w(int layer)  { return off_ln1_b(layer) + DIM; }
__host__ __device__ inline int off_qkv_b(int layer)  { return off_qkv_w(layer) + DIM*3*DIM; }
__host__ __device__ inline int off_attn_o_w(int layer) { return off_qkv_b(layer) + 3*DIM; }
__host__ __device__ inline int off_attn_o_b(int layer) { return off_attn_o_w(layer) + DIM*DIM; }
__host__ __device__ inline int off_ln2_w(int layer)  { return off_attn_o_b(layer) + DIM; }
__host__ __device__ inline int off_ln2_b(int layer)  { return off_ln2_w(layer) + DIM; }
__host__ __device__ inline int off_fc1_w(int layer)  { return off_ln2_b(layer) + DIM; }
__host__ __device__ inline int off_fc1_b(int layer)  { return off_fc1_w(layer) + DIM*FFN_DIM; }
__host__ __device__ inline int off_fc2_w(int layer)  { return off_fc1_b(layer) + FFN_DIM; }
__host__ __device__ inline int off_fc2_b(int layer)  { return off_fc2_w(layer) + FFN_DIM*DIM; }

#define KV_CACHE_LAYER_FLOATS (N_HEADS * MAX_SEQ_LEN * HEAD_DIM)

__device__ inline int kv_cache_len_offset(int layer) { return layer; }

__device__ inline int kv_cache_offset(int layer, int head, int pos, int d) {
    return layer * KV_CACHE_LAYER_FLOATS + head * (MAX_SEQ_LEN * HEAD_DIM) + pos * HEAD_DIM + d;
}

// ---------------------------------------------------------------------------
// Device helper: GELU activation
// ---------------------------------------------------------------------------
__device__ inline float gelu_fwd(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------
// Device helper: GEMV
// ---------------------------------------------------------------------------
__device__ void gemv(float* y, const float* W, const float* x,
                     const float* bias, int M, int N) {
    int tid = threadIdx.x;
    for (int i = tid; i < M; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += W[i * (long long)N + j] * x[j];
        }
        y[i] = sum + (bias ? bias[i] : 0.0f);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Device helper: LayerNorm
// ---------------------------------------------------------------------------
__device__ void layer_norm(float* x, const float* w, const float* b,
                            int dim, float eps) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < dim; i += nthreads) sum += x[i];

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* vs = sdata;
        if (nthreads >= 64) vs[tid] += vs[tid + 32];
        if (nthreads >= 32) vs[tid] += vs[tid + 16];
        if (nthreads >= 16) vs[tid] += vs[tid + 8];
        if (nthreads >= 8)  vs[tid] += vs[tid + 4];
        if (nthreads >= 4)  vs[tid] += vs[tid + 2];
        if (nthreads >= 2)  vs[tid] += vs[tid + 1];
    }
    __syncthreads();
    float mean = sdata[0] / dim;

    float var_sum = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        float d = x[i] - mean;
        var_sum += d * d;
    }
    sdata[tid] = var_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* vs = sdata;
        if (nthreads >= 64) vs[tid] += vs[tid + 32];
        if (nthreads >= 32) vs[tid] += vs[tid + 16];
        if (nthreads >= 16) vs[tid] += vs[tid + 8];
        if (nthreads >= 8)  vs[tid] += vs[tid + 4];
        if (nthreads >= 4)  vs[tid] += vs[tid + 2];
        if (nthreads >= 2)  vs[tid] += vs[tid + 1];
    }
    __syncthreads();
    float var = sdata[0] / dim;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < dim; i += nthreads) {
        x[i] = (x[i] - mean) * inv_std * w[i] + (b ? b[i] : 0.0f);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Device: Multi-head attention for ONE head
// ---------------------------------------------------------------------------
__device__ void attention_head(
    const float* q_head, const float* k_head, const float* v_head,
    float* k_cache_head, float* v_cache_head,
    int old_seq_len, float* out_head
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int new_len = old_seq_len + 1;

    int write_pos = old_seq_len;
    for (int d = tid; d < HEAD_DIM; d += nthreads) {
        k_cache_head[write_pos * HEAD_DIM + d] = k_head[d];
        v_cache_head[write_pos * HEAD_DIM + d] = v_head[d];
    }
    __syncthreads();

    __shared__ float scores[MAX_SEQ_LEN];
    for (int pos = tid; pos < new_len; pos += nthreads) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            s += q_head[d] * k_cache_head[pos * HEAD_DIM + d];
        scores[pos] = s * rsqrtf((float)HEAD_DIM);
    }
    __syncthreads();

    float my_max = -1e30f;
    for (int pos = tid; pos < new_len; pos += nthreads)
        if (scores[pos] > my_max) my_max = scores[pos];

    __shared__ float smax;
    __shared__ float ssum;
    {
        __shared__ float rbuf[256];
        rbuf[tid] = my_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 32; s >>= 1) {
            if (tid < s) rbuf[tid] = fmaxf(rbuf[tid], rbuf[tid + s]);
            __syncthreads();
        }
        if (tid < 32) {
            volatile float* vs = rbuf;
            if (nthreads >= 64) vs[tid] = fmaxf(vs[tid], vs[tid + 32]);
            if (nthreads >= 32) vs[tid] = fmaxf(vs[tid], vs[tid + 16]);
            if (nthreads >= 16) vs[tid] = fmaxf(vs[tid], vs[tid + 8]);
            if (nthreads >= 8)  vs[tid] = fmaxf(vs[tid], vs[tid + 4]);
            if (nthreads >= 4)  vs[tid] = fmaxf(vs[tid], vs[tid + 2]);
            if (nthreads >= 2)  vs[tid] = fmaxf(vs[tid], vs[tid + 1]);
        }
        if (tid == 0) smax = rbuf[0];
        __syncthreads();
    }

    float my_sum = 0.0f;
    for (int pos = tid; pos < new_len; pos += nthreads) {
        scores[pos] = expf(scores[pos] - smax);
        my_sum += scores[pos];
    }
    __syncthreads();

    {
        __shared__ float rbuf[256];
        rbuf[tid] = my_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 32; s >>= 1) {
            if (tid < s) rbuf[tid] += rbuf[tid + s];
            __syncthreads();
        }
        if (tid < 32) {
            volatile float* vs = rbuf;
            if (nthreads >= 64) vs[tid] += vs[tid + 32];
            if (nthreads >= 32) vs[tid] += vs[tid + 16];
            if (nthreads >= 16) vs[tid] += vs[tid + 8];
            if (nthreads >= 8)  vs[tid] += vs[tid + 4];
            if (nthreads >= 4)  vs[tid] += vs[tid + 2];
            if (nthreads >= 2)  vs[tid] += vs[tid + 1];
        }
        if (tid == 0) ssum = rbuf[0];
        __syncthreads();
    }

    float inv_sum = 1.0f / ssum;
    for (int pos = tid; pos < new_len; pos += nthreads)
        scores[pos] *= inv_sum;
    __syncthreads();

    for (int d = tid; d < HEAD_DIM; d += nthreads) {
        float acc = 0.0f;
        for (int pos = 0; pos < new_len; pos++)
            acc += scores[pos] * v_cache_head[pos * HEAD_DIM + d];
        out_head[d] = acc;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Shared-memory workspace for one decode step
// ---------------------------------------------------------------------------
struct DecodeWorkspace {
    float h[DIM];
    float residual[DIM];
    float normed[DIM];
    float q[DIM];
    float k[DIM];
    float v[DIM];
    float attn_out[DIM];
    float qkv[3 * DIM];
    float attn_proj[DIM];
    float fc1_out[FFN_DIM];
    float fc2_out[DIM];
    float q_head[HEAD_DIM];
    float k_head[HEAD_DIM];
    float v_head[HEAD_DIM];
    float out_head[HEAD_DIM];
};

// ---------------------------------------------------------------------------
// Device-side timing accumulator (clock64 cycles)
// ---------------------------------------------------------------------------
struct KernelStats {
    unsigned long long compute_cycles_total;
    int32_t            n_timed_steps;
    int32_t            _pad;
};

// ---------------------------------------------------------------------------
// Device: Full decode segment
// ---------------------------------------------------------------------------
__device__ void decode_segment(
    const float* input, float* output, float* weights,
    float* k_cache, float* v_cache, int* kv_len,
    DecodeWorkspace* ws
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    float* h = ws->h;

    for (int i = tid; i < DIM; i += nthreads) h[i] = input[i];
    __syncthreads();

    for (int layer = 0; layer < N_LAYERS; layer++) {
        for (int i = tid; i < DIM; i += nthreads) ws->residual[i] = h[i];
        __syncthreads();

        for (int i = tid; i < DIM; i += nthreads) ws->normed[i] = h[i];
        __syncthreads();
        layer_norm(ws->normed, weights + off_ln1_w(layer),
                   weights + off_ln1_b(layer), DIM, EPS);

        gemv(ws->qkv, weights + off_qkv_w(layer), ws->normed,
             weights + off_qkv_b(layer), 3 * DIM, DIM);
        for (int i = tid; i < DIM; i += nthreads) {
            ws->q[i] = ws->qkv[i];
            ws->k[i] = ws->qkv[DIM + i];
            ws->v[i] = ws->qkv[2 * DIM + i];
        }
        __syncthreads();

        for (int i = tid; i < DIM; i += nthreads) ws->attn_out[i] = 0.0f;
        __syncthreads();

        int seq_len = kv_len[layer];

        for (int hd = 0; hd < N_HEADS; hd++) {
            int hd_off = hd * HEAD_DIM;
            for (int d = tid; d < HEAD_DIM; d += nthreads) {
                ws->q_head[d] = ws->q[hd_off + d];
                ws->k_head[d] = ws->k[hd_off + d];
                ws->v_head[d] = ws->v[hd_off + d];
            }
            __syncthreads();

            attention_head(
                ws->q_head, ws->k_head, ws->v_head,
                k_cache + layer * KV_CACHE_LAYER_FLOATS + hd * (MAX_SEQ_LEN * HEAD_DIM),
                v_cache + layer * KV_CACHE_LAYER_FLOATS + hd * (MAX_SEQ_LEN * HEAD_DIM),
                seq_len, ws->out_head
            );

            for (int d = tid; d < HEAD_DIM; d += nthreads)
                ws->attn_out[hd_off + d] = ws->out_head[d];
            __syncthreads();
        }

        if (tid == 0) kv_len[layer] = seq_len + 1;
        __syncthreads();

        gemv(ws->attn_proj, weights + off_attn_o_w(layer), ws->attn_out,
             weights + off_attn_o_b(layer), DIM, DIM);
        for (int i = tid; i < DIM; i += nthreads)
            h[i] = ws->residual[i] + ws->attn_proj[i];
        __syncthreads();

        for (int i = tid; i < DIM; i += nthreads) ws->residual[i] = h[i];
        __syncthreads();
        for (int i = tid; i < DIM; i += nthreads) ws->normed[i] = h[i];
        __syncthreads();
        layer_norm(ws->normed, weights + off_ln2_w(layer),
                   weights + off_ln2_b(layer), DIM, EPS);

        gemv(ws->fc1_out, weights + off_fc1_w(layer), ws->normed,
             weights + off_fc1_b(layer), FFN_DIM, DIM);
        for (int i = tid; i < FFN_DIM; i += nthreads)
            ws->fc1_out[i] = gelu_fwd(ws->fc1_out[i]);
        __syncthreads();

        gemv(ws->fc2_out, weights + off_fc2_w(layer), ws->fc1_out,
             weights + off_fc2_b(layer), DIM, FFN_DIM);
        for (int i = tid; i < DIM; i += nthreads)
            h[i] = ws->residual[i] + ws->fc2_out[i];
        __syncthreads();
    }

    for (int i = tid; i < DIM; i += nthreads) output[i] = h[i];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Fused single-step kernel with optional clock64 instrumentation
// ---------------------------------------------------------------------------
__global__ void transformer_step_kernel(
    const float* input, float* output, float* weights,
    float* k_cache, float* v_cache, int* kv_len,
    KernelStats* stats
) {
    __shared__ DecodeWorkspace ws;

    unsigned long long t_start = 0;
    if (stats) t_start = clock64();

    decode_segment(input, output, weights, k_cache, v_cache, kv_len, &ws);

    if (stats && threadIdx.x == 0) {
        unsigned long long t_end = clock64();
        atomicAdd((unsigned long long*)&stats->compute_cycles_total, t_end - t_start);
        atomicAdd((int*)&stats->n_timed_steps, 1);
    }
}

// ---------------------------------------------------------------------------
// No-op kernel with the same launch signature (for launch-only overhead)
// ---------------------------------------------------------------------------
__global__ void noop_kernel(
    const float* input, float* output, float* weights,
    float* k_cache, float* v_cache, int* kv_len,
    KernelStats* stats
) {
    if (stats && threadIdx.x == 0) {
        atomicAdd((int*)&stats->n_timed_steps, 1);
    }
}

// ---------------------------------------------------------------------------
// Host: weight initialization
// ---------------------------------------------------------------------------
void init_weights(float* h_weights, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    int n_floats = N_LAYERS * LAYER_FLOATS;
    for (int i = 0; i < n_floats; i++) h_weights[i] = dist(rng);
    for (int layer = 0; layer < N_LAYERS; layer++) {
        float* ln1_w = h_weights + off_ln1_w(layer);
        float* ln1_b = h_weights + off_ln1_b(layer);
        float* ln2_w = h_weights + off_ln2_w(layer);
        float* ln2_b = h_weights + off_ln2_b(layer);
        for (int i = 0; i < DIM; i++) {
            ln1_w[i] = 1.0f; ln1_b[i] = 0.0f;
            ln2_w[i] = 1.0f; ln2_b[i] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------
struct Stats {
    float mean;
    float std;
    float median;
    float p25;
    float p75;
    float min;
    float max;
};

Stats compute_stats(std::vector<float>& data) {
    Stats s;
    size_t n = data.size();
    if (n == 0) { s.mean = s.std = s.median = s.p25 = s.p75 = s.min = s.max = 0; return s; }

    double sum = 0.0;
    for (auto v : data) sum += v;
    s.mean = (float)(sum / n);

    double var = 0.0;
    for (auto v : data) {
        double d = v - s.mean;
        var += d * d;
    }
    s.std = (float)sqrt(var / (n > 1 ? n - 1 : 1));

    std::sort(data.begin(), data.end());
    s.min = data.front();
    s.max = data.back();
    s.median = (n % 2 == 0) ? (data[n/2 - 1] + data[n/2]) * 0.5f : data[n/2];
    s.p25 = data[n/4];
    s.p75 = data[3*n/4];

    return s;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int N = 500;  // must stay < MAX_SEQ_LEN (512) so KV cache doesn't overflow
    bool verbose = false;
    const char* json_out = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--launches") == 0 && i + 1 < argc)
            N = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            verbose = true;
        else if (strcmp(argv[i], "--json-out") == 0 && i + 1 < argc)
            json_out = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: kappa_fused_direct [--launches N] [--json-out path] [--verbose]\n");
            return 0;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
    double cycles_per_ms = (double)clock_rate_khz;

    printf("=== κ_fused direct measurement ===\n");
    printf("  Device: %s (sm_%d%d), clockRate=%d kHz\n",
           prop.name, prop.major, prop.minor, clock_rate_khz);
    printf("  Model: %d layers, dim=%d, %d heads, FFN=%d\n",
           N_LAYERS, DIM, N_HEADS, FFN_DIM);
    printf("  Launches: %d\n", N);

    int weights_floats = N_LAYERS * LAYER_FLOATS;
    int kv_floats_total = N_LAYERS * N_HEADS * MAX_SEQ_LEN * HEAD_DIM;

    std::vector<float> h_weights(weights_floats);
    std::vector<float> h_k_cache(kv_floats_total, 0.0f);
    std::vector<float> h_v_cache(kv_floats_total, 0.0f);
    std::vector<int>   h_kv_len(N_LAYERS, 0);
    init_weights(h_weights.data());

    float *d_weights, *d_k_cache, *d_v_cache;
    int *d_kv_len;
    float *d_input, *d_output;
    KernelStats *d_stats;
    KernelStats h_stats;

    CUDA_CHECK(cudaMalloc(&d_weights, weights_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kv_len, N_LAYERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(KernelStats)));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weights_floats * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), kv_floats_total * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), kv_floats_total * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kv_len, h_kv_len.data(), N_LAYERS * sizeof(int),
               cudaMemcpyHostToDevice));

    std::vector<float> h_input(DIM);
    for (int i = 0; i < DIM; i++) h_input[i] = 0.01f;
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), DIM * sizeof(float),
               cudaMemcpyHostToDevice));

    // Reset KV cache lengths for fresh sequence
    std::vector<int> zero_len(N_LAYERS, 0);
    CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
               cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // ── Warm-up: 2 batches of N launches (stabilize GPU boost clock) ──
    // Each batch resets KV cache lengths to stay within MAX_SEQ_LEN.
    // Sync after each batch to catch errors early and avoid TDR.
    int n_warmup_batches = 3;
    for (int b = 0; b < n_warmup_batches; b++) {
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
                   cudaMemcpyHostToDevice));
        for (int i = 0; i < N; i++) {
            transformer_step_kernel<<<1, 256, 0, stream>>>(
                d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, nullptr);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (verbose) printf("  Warm-up batch %d done\n", b);
    }
    if (verbose) printf("  Warm-up done (%d x %d launches)\n", n_warmup_batches, N);

    // ── Ensure clean KV cache for measurement ─────────────────────────
    CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(KernelStats)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Measurement A: loop-level real kernel (single event pair) ─────
    // This measures N*(compute + κ_fused) with the lowest possible
    // measurement overhead: one event pair for the whole loop.
    // Per-step overhead from the event pair is event_cost/N, negligible
    // for N >= 1000.
    cudaEvent_t lr_start, lr_end;
    CUDA_CHECK(cudaEventCreate(&lr_start));
    CUDA_CHECK(cudaEventCreate(&lr_end));

    CUDA_CHECK(cudaEventRecord(lr_start, stream));
    for (int i = 0; i < N; i++) {
        transformer_step_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, d_stats);
    }
    CUDA_CHECK(cudaEventRecord(lr_end, stream));
    CUDA_CHECK(cudaEventSynchronize(lr_end));

    float real_loop_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&real_loop_total_ms, lr_start, lr_end));
    double real_per_step_ms = real_loop_total_ms / N;

    CUDA_CHECK(cudaMemcpy(&h_stats, d_stats, sizeof(KernelStats),
               cudaMemcpyDeviceToHost));
    double compute_cycles_per_step = (h_stats.n_timed_steps > 0)
        ? (double)h_stats.compute_cycles_total / h_stats.n_timed_steps
        : 0.0;
    double compute_ms_at_base = compute_cycles_per_step / cycles_per_ms;

    if (verbose) {
        printf("\n  Real kernel loop (N=%d, single event pair):\n", N);
        printf("    total = %.3f ms, per_step = %.6f ms\n",
               real_loop_total_ms, real_per_step_ms);
        printf("    clock64 compute: %.0f cycles/step = %.6f ms (at base)\n",
               compute_cycles_per_step, compute_ms_at_base);
    }

    // ── Measurement B: loop-level noop kernel (single event pair) ─────
    // The noop kernel does nothing but return. CUDA event time for a
    // loop of N noop launches = N*κ_fused + event_overhead.
    // κ_fused = noop_loop_total / N - event_overhead / N.
    // For N=1000, event_overhead/N is negligible (~0.001 us).
    cudaEvent_t ln_start, ln_end;
    CUDA_CHECK(cudaEventCreate(&ln_start));
    CUDA_CHECK(cudaEventCreate(&ln_end));

    CUDA_CHECK(cudaEventRecord(ln_start, stream));
    for (int i = 0; i < N; i++) {
        noop_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(ln_end, stream));
    CUDA_CHECK(cudaEventSynchronize(ln_end));

    float noop_loop_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&noop_loop_total_ms, ln_start, ln_end));
    double noop_per_step_ms = noop_loop_total_ms / N;

    if (verbose) {
        printf("\n  Noop kernel loop (N=%d, single event pair):\n", N);
        printf("    total = %.6f ms, per_step = %.6f ms\n",
               noop_loop_total_ms, noop_per_step_ms);
    }

    // ── Measurement B2: forced-sync noop kernel (single event pair) ────
    // Same noop loop, but call cudaStreamSynchronize() after EVERY launch.
    // This defeats WDDM command-buffer batching: each launch is fully
    // dispatched and completed before the next begins.
    // The total includes full GPU round-trip sync overhead per launch.
    cudaEvent_t lns_start, lns_end;
    CUDA_CHECK(cudaEventCreate(&lns_start));
    CUDA_CHECK(cudaEventCreate(&lns_end));

    CUDA_CHECK(cudaEventRecord(lns_start, stream));
    for (int i = 0; i < N; i++) {
        noop_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaEventRecord(lns_end, stream));
    CUDA_CHECK(cudaEventSynchronize(lns_end));

    float noop_sync_loop_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&noop_sync_loop_total_ms, lns_start, lns_end));
    double noop_sync_per_step_ms = noop_sync_loop_total_ms / N;

    // Also record per-launch times for the forced-sync noop variant
    // to get distribution stats (median, IQR) with sync.
    int N_sync_per = std::min(N, 2000);
    std::vector<cudaEvent_t> pns_start(N_sync_per), pns_end(N_sync_per);
    for (int i = 0; i < N_sync_per; i++) {
        CUDA_CHECK(cudaEventCreate(&pns_start[i]));
        CUDA_CHECK(cudaEventCreate(&pns_end[i]));
    }
    for (int i = 0; i < N_sync_per; i++) {
        CUDA_CHECK(cudaEventRecord(pns_start[i], stream));
        noop_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(pns_end[i], stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> per_noop_sync_ms(N_sync_per);
    for (int i = 0; i < N_sync_per; i++) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pns_start[i], pns_end[i]));
        per_noop_sync_ms[i] = ms;
    }
    for (int i = 0; i < N_sync_per; i++) {
        CUDA_CHECK(cudaEventDestroy(pns_start[i]));
        CUDA_CHECK(cudaEventDestroy(pns_end[i]));
    }
    Stats s_noop_sync = compute_stats(per_noop_sync_ms);

    CUDA_CHECK(cudaEventDestroy(lns_start));
    CUDA_CHECK(cudaEventDestroy(lns_end));

    if (verbose) {
        printf("\n  Noop kernel FORCED SYNC (N=%d, stream sync after each launch):\n", N);
        printf("    loop total = %.6f ms, per_step = %.6f ms\n",
               noop_sync_loop_total_ms, noop_sync_per_step_ms);
        printf("    per-launch stats (N=%d):\n", N_sync_per);
        printf("      median = %.6f ms, mean = %.6f ms, std = %.6f ms\n",
               s_noop_sync.median, s_noop_sync.mean, s_noop_sync.std);
        printf("      p25 = %.6f ms, p75 = %.6f ms\n",
               s_noop_sync.p25, s_noop_sync.p75);
        printf("      std/mean = %.4f\n",
               s_noop_sync.std / (s_noop_sync.mean > 0 ? s_noop_sync.mean : 1));
    }

    // ── Measurement C: per-launch event pairs (for distribution) ──────
    // Real kernel per-launch times — gives median, IQR, min, max.
    // Use a fresh KV cache segment so seq_len stays small.
    CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(KernelStats)));

    int N_per_real = std::min(N, 100);  // profile fewer launches for real kernel per-launch dist
    std::vector<cudaEvent_t> ps_start(N_per_real), ps_end(N_per_real);
    for (int i = 0; i < N_per_real; i++) {
        CUDA_CHECK(cudaEventCreate(&ps_start[i]));
        CUDA_CHECK(cudaEventCreate(&ps_end[i]));
    }

    for (int i = 0; i < N_per_real; i++) {
        CUDA_CHECK(cudaEventRecord(ps_start[i], stream));
        transformer_step_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, d_stats);
        CUDA_CHECK(cudaEventRecord(ps_end[i], stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> per_real_ms(N_per_real);
    for (int i = 0; i < N_per_real; i++) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ps_start[i], ps_end[i]));
        per_real_ms[i] = ms;
    }
    for (int i = 0; i < N_per_real; i++) {
        CUDA_CHECK(cudaEventDestroy(ps_start[i]));
        CUDA_CHECK(cudaEventDestroy(ps_end[i]));
    }
    Stats s_real = compute_stats(per_real_ms);

    // Read back clock64 from the per-launch batch too
    CUDA_CHECK(cudaMemcpy(&h_stats, d_stats, sizeof(KernelStats),
               cudaMemcpyDeviceToHost));

    if (verbose) {
        printf("\n  Real kernel per-launch (N=%d, CUDA event pairs):\n", N_per_real);
        printf("    median = %.6f ms, mean = %.6f ms, std = %.6f ms\n",
               s_real.median, s_real.mean, s_real.std);
        printf("    p25 = %.6f ms, p75 = %.6f ms\n", s_real.p25, s_real.p75);
        printf("    min = %.6f ms, max = %.6f ms\n", s_real.min, s_real.max);
    }

    // Noop per-launch times — use many more samples for a stable median
    int N_per_noop = 2000;  // 20x more than the original 100
    std::vector<cudaEvent_t> pn_start(N_per_noop), pn_end(N_per_noop);
    for (int i = 0; i < N_per_noop; i++) {
        CUDA_CHECK(cudaEventCreate(&pn_start[i]));
        CUDA_CHECK(cudaEventCreate(&pn_end[i]));
    }

    for (int i = 0; i < N_per_noop; i++) {
        CUDA_CHECK(cudaEventRecord(pn_start[i], stream));
        noop_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len, nullptr);
        CUDA_CHECK(cudaEventRecord(pn_end[i], stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> per_noop_ms(N_per_noop);
    for (int i = 0; i < N_per_noop; i++) {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, pn_start[i], pn_end[i]));
        per_noop_ms[i] = ms;
    }
    for (int i = 0; i < N_per_noop; i++) {
        CUDA_CHECK(cudaEventDestroy(pn_start[i]));
        CUDA_CHECK(cudaEventDestroy(pn_end[i]));
    }
    Stats s_noop = compute_stats(per_noop_ms);

    if (verbose) {
        printf("  Noop kernel per-launch (N=%d, CUDA event pairs):\n", N_per_noop);
        printf("    median = %.6f ms, mean = %.6f ms, std = %.6f ms\n",
               s_noop.median, s_noop.mean, s_noop.std);
        printf("    p25 = %.6f ms, p75 = %.6f ms\n", s_noop.p25, s_noop.p75);
        printf("    std/mean = %.4f\n", s_noop.std / (s_noop.mean > 0 ? s_noop.mean : 1));
    }

    // ── Derived κ_fused ──────────────────────────────────────────────
    // Primary: noop loop-level (single event pair, divided by N).
    //   κ_fused = noop_loop_total / N  (event overhead negligible for N>>1)
    // This is the most direct measurement: we launch a noop kernel with the
    // same launch signature N times, measure the total wall time, and
    // divide by N.
    double kappa_fused_ms = noop_per_step_ms;
    double kappa_fused_us = kappa_fused_ms * 1000.0;

    // Cross-check: real_loop_per_step - κ_fused should ≈ compute (CUDA event)
    // (but real_loop_per_step includes compute + κ_fused, and compute via
    // clock64 needs a boost adjustment to convert cycles to wall time).
    double compute_cuda_event_ms = real_per_step_ms - kappa_fused_ms;

    // Exit early: κ_fused is the key measurement. The stdout report below
    // provides the full picture.

    // ── Output ──────────────────────────────────────────────────────
    printf("\n=== Results ===\n");
    printf("  Real kernel loop:       total = %.3f ms, per_step = %.6f ms\n",
           real_loop_total_ms, real_per_step_ms);
    printf("\n--- No-sync (WDDM batched) ---\n");
    printf("  Noop loop:              total = %.6f ms, per_step = %.6f ms\n",
           noop_loop_total_ms, noop_per_step_ms);
    printf("  κ_fused (noop loop / N):        %.6f ms  (%.3f us)\n",
           kappa_fused_ms, kappa_fused_us);
    printf("  Noop per-launch median  (N=%d): %.6f ms\n", N_per_noop, s_noop.median);
    printf("  Noop per-launch mean    (N=%d): %.6f ms\n", N_per_noop, s_noop.mean);
    printf("  Noop per-launch std     (N=%d): %.6f ms\n", N_per_noop, s_noop.std);
    printf("  Noop per-launch std/mean:       %.4f\n",
           s_noop.std / (s_noop.mean > 0 ? s_noop.mean : 1));
    printf("  Noop per-launch p25/p75:        %.6f / %.6f ms\n", s_noop.p25, s_noop.p75);
    printf("\n--- Forced-sync (per-launch stream sync) ---\n");
    printf("  Noop sync loop:         total = %.6f ms, per_step = %.6f ms\n",
           noop_sync_loop_total_ms, noop_sync_per_step_ms);
    printf("  κ_fused_sync (sync loop / N):   %.6f ms  (%.3f us)\n",
           noop_sync_per_step_ms, noop_sync_per_step_ms * 1000.0);
    printf("  Noop sync per-launch median:    %.6f ms\n", s_noop_sync.median);
    printf("  Noop sync per-launch mean:      %.6f ms\n", s_noop_sync.mean);
    printf("  Noop sync per-launch std:       %.6f ms\n", s_noop_sync.std);
    printf("  Noop sync per-launch std/mean:  %.4f\n",
           s_noop_sync.std / (s_noop_sync.mean > 0 ? s_noop_sync.mean : 1));
    printf("  Noop sync per-launch p25/p75:   %.6f / %.6f ms\n", s_noop_sync.p25, s_noop_sync.p75);
    printf("\n--- Derived ---\n");
    printf("  real loop - κ_fused (compute):  %.6f ms\n", compute_cuda_event_ms);
    printf("  clock64 compute (at base):      %.6f ms  (%s)\n",
           compute_ms_at_base, verbose ? "" : "use --verbose for cycles");
    printf("  clock64 cycles/step:            %.0f\n", compute_cycles_per_step);
    printf("  Real per-launch median (N=%d):  %.6f ms\n", N_per_real, s_real.median);

    CUDA_CHECK(cudaEventDestroy(lr_start));
    CUDA_CHECK(cudaEventDestroy(lr_end));
    CUDA_CHECK(cudaEventDestroy(ln_start));
    CUDA_CHECK(cudaEventDestroy(ln_end));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_kv_len));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_stats));

    if (json_out) {
        FILE* f = fopen(json_out, "w");
        if (f) {
            fprintf(f,
                "{\n"
                "  \"device\": \"%s\",\n"
                "  \"sm\": \"%d%d\",\n"
                "  \"clock_rate_khz\": %d,\n"
                "  \"model\": {\"n_layers\": %d, \"dim\": %d, \"n_heads\": %d, \"ffn_dim\": %d},\n"
                "  \"n_warmup_batches\": %d,\n"
                "  \"n_warmup_per_batch\": %d,\n"
                "  \"n_loop_launches\": %d,\n"
                "  \"n_per_launch_real\": %d,\n"
                "  \"n_per_launch_noop\": %d,\n"
                "  \"real_loop_total_ms\": %.8f,\n"
                "  \"real_per_step_ms\": %.8f,\n"
                "  \"noop_loop_total_ms\": %.8f,\n"
                "  \"noop_per_step_ms\": %.8f,\n"
                "  \"kappa_fused_ms\": %.8f,\n"
                "  \"kappa_fused_us\": %.4f,\n"
                "  \"kappa_source\": \"noop_loop_single_event_pair_div_N\",\n"
                "  \"compute_cuda_event_ms\": %.8f,\n"
                "  \"compute_cuda_event_source\": \"real_per_step_minus_kappa\",\n"
                "  \"clock64_compute_cycles_per_step\": %.0f,\n"
                "  \"clock64_compute_ms_at_base_clock\": %.8f,\n"
                "  \"clock64_n_timed_steps\": %d,\n"
                "  \"per_launch_real_ms\": {\n"
                "    \"median\": %.8f,\n"
                "    \"mean\": %.8f,\n"
                "    \"std\": %.8f,\n"
                "    \"p25\": %.8f,\n"
                "    \"p75\": %.8f,\n"
                "    \"min\": %.8f,\n"
                "    \"max\": %.8f,\n"
                "    \"unit\": \"ms\"\n"
                "  },\n"
                "  \"per_launch_noop_ms\": {\n"
                "    \"median\": %.8f,\n"
                "    \"mean\": %.8f,\n"
                "    \"std\": %.8f,\n"
                "    \"p25\": %.8f,\n"
                "    \"p75\": %.8f,\n"
                "    \"min\": %.8f,\n"
                "    \"max\": %.8f,\n"
                "    \"unit\": \"ms\"\n"
                "  },\n"
                "  \"noop_sync_loop_total_ms\": %.8f,\n"
                "  \"noop_sync_per_step_ms\": %.8f,\n"
                "  \"kappa_fused_sync_ms\": %.8f,\n"
                "  \"kappa_fused_sync_us\": %.4f,\n"
                "  \"per_launch_noop_sync_ms\": {\n"
                "    \"median\": %.8f,\n"
                "    \"mean\": %.8f,\n"
                "    \"std\": %.8f,\n"
                "    \"p25\": %.8f,\n"
                "    \"p75\": %.8f,\n"
                "    \"min\": %.8f,\n"
                "    \"max\": %.8f,\n"
                "    \"unit\": \"ms\"\n"
                "  }\n"
                "}\n",
                prop.name, prop.major, prop.minor, clock_rate_khz,
                N_LAYERS, DIM, N_HEADS, FFN_DIM,
                n_warmup_batches, N,
                N, N_per_real, N_per_noop,
                real_loop_total_ms, real_per_step_ms,
                noop_loop_total_ms, noop_per_step_ms,
                kappa_fused_ms, kappa_fused_us,
                compute_cuda_event_ms,
                compute_cycles_per_step, compute_ms_at_base,
                h_stats.n_timed_steps,
                s_real.median, s_real.mean, s_real.std,
                s_real.p25, s_real.p75, s_real.min, s_real.max,
                s_noop.median, s_noop.mean, s_noop.std,
                s_noop.p25, s_noop.p75, s_noop.min, s_noop.max,
                noop_sync_loop_total_ms, noop_sync_per_step_ms,
                noop_sync_per_step_ms, noop_sync_per_step_ms * 1000.0,
                s_noop_sync.median, s_noop_sync.mean, s_noop_sync.std,
                s_noop_sync.p25, s_noop_sync.p75, s_noop_sync.min, s_noop_sync.max);
            fclose(f);
            printf("\nWrote %s\n", json_out);
        } else {
            fprintf(stderr, "WARNING: could not write %s\n", json_out);
        }
    }

    return 0;
}
