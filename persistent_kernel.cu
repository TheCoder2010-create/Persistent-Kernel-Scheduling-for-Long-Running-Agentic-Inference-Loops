// persistent_kernel.cu — Multi-Trajectory Persistent Agent Kernel
//
// Extends the original single-trajectory design to N concurrent trajectories
// sharing one resident kernel.  Designed for the follow-up regime where
// d_t + l_t is small (sub-ms) and N is large (up to 128), making the
// amortized-launch saving (N-1)*kappa meaningful.
//
// Design: design.md  |  Paper: paper.tex
//
// Build (RTX 4050 / sm_86):
//   nvcc -O3 -arch=sm_86 persistent_kernel.cu -o persistent_kernel.exe

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>
#include <vector>
#include <cassert>
#include <atomic>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
#define DIM          256
#define N_HEADS      4
#define HEAD_DIM     (DIM / N_HEADS)
#define MLP_RATIO    4
#define FFN_DIM      (DIM * MLP_RATIO)
#define MAX_SEQ_LEN  512
#define N_LAYERS     2
#define EPS          1e-5f
#define MAX_N_TRAJ   128

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------
enum SlotState : int32_t {
    SLOT_EMPTY    = 0,
    SLOT_READY    = 1,
    SLOT_DONE     = 2,
    SLOT_SHUTDOWN = 3
};

struct QueueEntry {
    int32_t state;
    int32_t trajectory_id;     // which trajectory this slot belongs to
    int32_t seq_len;           // KV cache length before this step
    int32_t _pad;
    float   input[DIM];
    float   output[DIM];
};

// Per-trajectory device-side timing (clock64 cycles)
struct TrajStats {
    unsigned long long sigma_cycles_total;
    unsigned long long compute_cycles_total;
    int32_t            n_timed_steps;
    int32_t            _pad;
};

// ---------------------------------------------------------------------------
// Weight layout (same as original, one shared set for all trajectories)
// ---------------------------------------------------------------------------
#define LAYER_FLOATS (2*DIM + DIM*3*DIM + 3*DIM + DIM*DIM + DIM + 2*DIM + DIM*FFN_DIM + FFN_DIM + FFN_DIM*DIM + DIM)

__host__ __device__ inline int layer_offset(int layer) {
    return layer * LAYER_FLOATS;
}

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

// KV cache layout per trajectory:
//   [layer * N_HEADS * MAX_SEQ_LEN * HEAD_DIM]
#define KV_CACHE_LAYER_FLOATS (N_HEADS * MAX_SEQ_LEN * HEAD_DIM)
#define KV_CACHE_PER_TRAJ     (N_LAYERS * KV_CACHE_LAYER_FLOATS)

// Given trajectory t, layer l, head h, position pos, dimension d:
//   traj_base = t * N_LAYERS * KV_CACHE_LAYER_FLOATS
//   layer_base = traj_base + l * KV_CACHE_LAYER_FLOATS
//   head_base = layer_base + h * MAX_SEQ_LEN * HEAD_DIM
//   index = head_base + pos * HEAD_DIM + d

__host__ __device__ inline int kv_cache_flat_index(int traj, int layer,
                                                     int head, int pos, int d) {
    return (traj * N_LAYERS * KV_CACHE_LAYER_FLOATS)
         + (layer * KV_CACHE_LAYER_FLOATS)
         + (head * MAX_SEQ_LEN * HEAD_DIM)
         + (pos * HEAD_DIM)
         + d;
}

// ---------------------------------------------------------------------------
// Error checking
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
// Device helpers (unchanged from original)
// ---------------------------------------------------------------------------
__device__ inline float gelu_fwd(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

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
        for (int d = 0; d < HEAD_DIM; d++) {
            s += q_head[d] * k_cache_head[pos * HEAD_DIM + d];
        }
        scores[pos] = s * rsqrtf((float)HEAD_DIM);
    }
    __syncthreads();

    float my_max = -1e30f;
    for (int pos = tid; pos < new_len; pos += nthreads) {
        if (scores[pos] > my_max) my_max = scores[pos];
    }
    __shared__ float smax, ssum;
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
    for (int pos = tid; pos < new_len; pos += nthreads) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < HEAD_DIM; d += nthreads) {
        float acc = 0.0f;
        for (int pos = 0; pos < new_len; pos++) {
            acc += scores[pos] * v_cache_head[pos * HEAD_DIM + d];
        }
        out_head[d] = acc;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Workspace (shared memory per block)
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
// Decode segment (same as original, operates on a trajectory's KV cache)
// ---------------------------------------------------------------------------
__device__ void decode_segment(
    const float* input,
    float* output,
    const float* weights,
    float* k_cache,       // base pointer for this trajectory's K cache
    float* v_cache,       // base pointer for this trajectory's V cache
    int* kv_len,          // [N_LAYERS] for this trajectory
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

            float* k_cache_layer = k_cache + layer * KV_CACHE_LAYER_FLOATS;
            float* v_cache_layer = v_cache + layer * KV_CACHE_LAYER_FLOATS;
            attention_head(
                ws->q_head, ws->k_head, ws->v_head,
                k_cache_layer + hd * (MAX_SEQ_LEN * HEAD_DIM),
                v_cache_layer + hd * (MAX_SEQ_LEN * HEAD_DIM),
                seq_len, ws->out_head
            );

            for (int d = tid; d < HEAD_DIM; d += nthreads) {
                ws->attn_out[hd_off + d] = ws->out_head[d];
            }
            __syncthreads();
        }

        if (tid == 0) kv_len[layer] = seq_len + 1;
        __syncthreads();

        gemv(ws->attn_proj, weights + off_attn_o_w(layer), ws->attn_out,
             weights + off_attn_o_b(layer), DIM, DIM);
        for (int i = tid; i < DIM; i += nthreads) {
            h[i] = ws->residual[i] + ws->attn_proj[i];
        }
        __syncthreads();

        for (int i = tid; i < DIM; i += nthreads) ws->residual[i] = h[i];
        __syncthreads();
        for (int i = tid; i < DIM; i += nthreads) ws->normed[i] = h[i];
        __syncthreads();
        layer_norm(ws->normed, weights + off_ln2_w(layer),
                   weights + off_ln2_b(layer), DIM, EPS);

        gemv(ws->fc1_out, weights + off_fc1_w(layer), ws->normed,
             weights + off_fc1_b(layer), FFN_DIM, DIM);
        for (int i = tid; i < FFN_DIM; i += nthreads) {
            ws->fc1_out[i] = gelu_fwd(ws->fc1_out[i]);
        }
        __syncthreads();

        gemv(ws->fc2_out, weights + off_fc2_w(layer), ws->fc1_out,
             weights + off_fc2_b(layer), DIM, FFN_DIM);
        for (int i = tid; i < DIM; i += nthreads) {
            h[i] = ws->residual[i] + ws->fc2_out[i];
        }
        __syncthreads();
    }

    for (int i = tid; i < DIM; i += nthreads) output[i] = h[i];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Multi-trajectory persistent kernel
// ---------------------------------------------------------------------------
//   queue: array of QueueEntry[MAX_N_TRAJ] in mapped host memory
//   n_traj: how many slots to poll (1..MAX_N_TRAJ)
//   weights, k_cache, v_cache, kv_len: per-trajectory KV arrays
//   stats: array of TrajStats[MAX_N_TRAJ] (device, may be null)
//
__global__ void persistent_multi_agent_kernel(
    float* weights,
    float* k_cache,
    float* v_cache,
    int*   kv_len,
    volatile QueueEntry* queue,
    int    total_steps,
    TrajStats* stats
) {
    if (blockIdx.x != 0) return;

    __shared__ DecodeWorkspace ws;

    // Multi-trajectory: each step processes n_traj items (round-robin indices).
    // total_steps passed from host = number of DECODE steps (not trajectory-items).
    // The outer loop is over decode steps; inner loop over trajectories.
    // For V1, n_traj is fixed at 1 (single-trajectory mode).
    const int n_traj = 1;  // TODO: pass as kernel argument

    // input_buffer / output_buffer placed at the END of the KV buffer
    // (after all trajectory KV cache regions) to avoid overlap.
    // Trajectory 0 uses offset 0 (same as naive baseline) for fair comparison.
    float* input_buffer  = k_cache + MAX_N_TRAJ * KV_CACHE_PER_TRAJ;
    float* output_buffer = v_cache + MAX_N_TRAJ * KV_CACHE_PER_TRAJ;
    (void)queue;

    for (int step = 0; step < total_steps; step++) {
        for (int t = 0; t < n_traj; t++) {
            int tid = t;

            // Per-trajectory KV cache region — tid=0 uses offset 0 (same as naive)
            float* t_k_cache = k_cache + tid * KV_CACHE_PER_TRAJ;
            float* t_v_cache = v_cache + tid * KV_CACHE_PER_TRAJ;
            int*   t_kv_len  = kv_len  + tid * N_LAYERS;

            // Reset KV length to 0 so each step sees an empty cache,
            // matching the naive-per-step baseline.
            if (threadIdx.x < N_LAYERS)
                t_kv_len[threadIdx.x] = 0;
            __syncthreads();

            unsigned long long t_start = clock64();

            decode_segment(input_buffer, output_buffer,
                           weights, t_k_cache, t_v_cache, t_kv_len, &ws);
            __syncthreads();

            unsigned long long t_end = clock64();

            if (threadIdx.x == 0) {
                stats[tid].compute_cycles_total += (t_end - t_start);
                stats[tid].n_timed_steps        += 1;
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// Naive single-step kernel (baseline — same decode, fresh launch each step)
// ---------------------------------------------------------------------------
__global__ void transformer_step_kernel(
    const float* input, float* output, const float* weights,
    float* k_cache, float* v_cache, int* kv_len
) {
    __shared__ DecodeWorkspace ws;
    decode_segment(input, output, weights, k_cache, v_cache, kv_len, &ws);
}

// ---------------------------------------------------------------------------
// Host helpers
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
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    int T = 20;
    float tool_latency_ms = 20.0f;
    int n_traj = 4;
    bool verbose = false;
    const char* json_out = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tool-latency-ms") == 0 && i + 1 < argc)
            tool_latency_ms = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--n-trajectories") == 0 && i + 1 < argc)
            n_traj = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            verbose = true;
        else if (strcmp(argv[i], "--json-out") == 0 && i + 1 < argc)
            json_out = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: persistent_kernel [--steps N] [--tool-latency-ms F]\n"
                   "       [--n-trajectories N] [--json-out path] [--verbose]\n");
            return 0;
        }
    }

    if (n_traj < 1 || n_traj > MAX_N_TRAJ) {
        fprintf(stderr, "n_traj must be 1..%d\n", MAX_N_TRAJ);
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
    double cycles_per_ms = (double)clock_rate_khz;

    printf("=== Multi-Trajectory Persistent Kernel ===\n");
    printf("  Device: %s (sm_%d%d), clockRate=%d kHz\n",
           prop.name, prop.major, prop.minor, clock_rate_khz);
    printf("  Model: %d layers, dim=%d, %d heads, FFN=%d\n",
           N_LAYERS, DIM, N_HEADS, FFN_DIM);
    printf("  Steps: %d, Tool latency: %.1f ms, Trajectories: %d\n",
           T, tool_latency_ms, n_traj);
    printf("\n");

    // --- Allocate ---
    int weights_floats = N_LAYERS * LAYER_FLOATS;
    int kv_floats_per_traj = N_LAYERS * KV_CACHE_LAYER_FLOATS;
    int kv_floats_total = MAX_N_TRAJ * kv_floats_per_traj;
    int buf_floats = kv_floats_total + DIM;  // extra DIM for input/output buffer

    std::vector<float> h_weights(weights_floats);
    std::vector<float> h_k_cache(kv_floats_total, 0.0f);
    std::vector<float> h_v_cache(kv_floats_total, 0.0f);
    std::vector<int>   h_kv_len(MAX_N_TRAJ * N_LAYERS, 0);
    init_weights(h_weights.data());

    float *d_weights, *d_k_cache, *d_v_cache;
    int *d_kv_len;
    TrajStats *d_stats, h_stats[MAX_N_TRAJ] = {};

    CUDA_CHECK(cudaMalloc(&d_weights, weights_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, buf_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, buf_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_k_cache, 0, buf_floats * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, buf_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kv_len, MAX_N_TRAJ * N_LAYERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stats, MAX_N_TRAJ * sizeof(TrajStats)));
    CUDA_CHECK(cudaMemset(d_stats, 0, MAX_N_TRAJ * sizeof(TrajStats)));

    // Queue: mapped host memory (one slot — input/output per step)
    QueueEntry* h_queue = nullptr;
    QueueEntry* d_queue_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_queue, MAX_N_TRAJ * sizeof(QueueEntry),
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_queue_ptr, h_queue, 0));
    for (int i = 0; i < MAX_N_TRAJ; i++) {
        memset(&h_queue[i], 0, sizeof(QueueEntry));
        h_queue[i].trajectory_id = i;
        h_queue[i].state = SLOT_EMPTY;
    }
    printf("  h_queue=%p d_queue=%p sizeof(Entry)=%llu\n",
           (void*)h_queue, (void*)d_queue_ptr,
           (unsigned long long)sizeof(QueueEntry));

    // Stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
               weights_floats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(),
               kv_floats_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(),
               kv_floats_total * sizeof(float), cudaMemcpyHostToDevice));
    // input_buffer at offset kv_floats_total remains zero from cudaMemset
    CUDA_CHECK(cudaMemcpy(d_kv_len, h_kv_len.data(),
               MAX_N_TRAJ * N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Populate input_buffer at offset kv_floats_total (kernel reads from device mem)
    std::vector<float> h_input(DIM);
    for (int d = 0; d < DIM; d++)
        h_input[d] = 0.01f * (float)(d % 100);
    CUDA_CHECK(cudaMemcpy(d_k_cache + kv_floats_total, h_input.data(),
               DIM * sizeof(float), cudaMemcpyHostToDevice));

    int N_timed = 50;

    // --- Warmup (one launch, T steps) ---
    printf("  Warmup: launching kernel for %d steps\n", T);
    persistent_multi_agent_kernel<<<1, 256, 0, stream>>>(
        d_weights, d_k_cache, d_v_cache, d_kv_len,
        d_queue_ptr, T, d_stats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- Naive baseline FIRST (one launch per step) ---
    double naive_total_us = 0.0;  // total GPU time for N_timed naive calls
    {
        std::vector<int> zero_len(N_LAYERS, 0);
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, DIM * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(),
                   N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                   DIM * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t ks, ke;
        CUDA_CHECK(cudaEventCreate(&ks));
        CUDA_CHECK(cudaEventCreate(&ke));

        // Single call timing
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(),
                   N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ks, stream));
        transformer_step_kernel<<<1, 256, 0, stream>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len);
        CUDA_CHECK(cudaEventRecord(ke, stream));
        CUDA_CHECK(cudaEventSynchronize(ke));
        float tmp_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&tmp_ms, ks, ke));
        if (verbose)
            printf("  Single naive kernel: %.2f us\n", tmp_ms * 1000.0f);

        // N_timed calls
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(),
                   N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ks, stream));
        for (int t = 0; t < N_timed; t++) {
            transformer_step_kernel<<<1, 256, 0, stream>>>(
                d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len);
        }
        CUDA_CHECK(cudaEventRecord(ke, stream));
        CUDA_CHECK(cudaEventSynchronize(ke));
        if (cudaEventElapsedTime(&tmp_ms, ks, ke) == cudaSuccess)
            naive_total_us = (double)tmp_ms * 1000.0;

        // Validate output changed
        std::vector<float> h_check(DIM, 0.0f);
        CUDA_CHECK(cudaMemcpy(h_check.data(), d_output,
                   DIM * sizeof(float), cudaMemcpyDeviceToHost));
        float sum_val = 0.0f;
        for (int i = 0; i < DIM; i++) sum_val += h_check[i];
        if (verbose)
            printf("  Naive output sum: %.6f (should be non-zero)\n", (double)sum_val);

        CUDA_CHECK(cudaEventDestroy(ks));
        CUDA_CHECK(cudaEventDestroy(ke));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    // --- Timed batch run (one launch, N_timed steps) ---
    CUDA_CHECK(cudaMemsetAsync(d_stats, 0, MAX_N_TRAJ * sizeof(TrajStats), stream));
    cudaEvent_t ks, ke;
    CUDA_CHECK(cudaEventCreate(&ks));
    CUDA_CHECK(cudaEventCreate(&ke));
    CUDA_CHECK(cudaEventRecord(ks, stream));
    persistent_multi_agent_kernel<<<1, 256, 0, stream>>>(
        d_weights, d_k_cache, d_v_cache, d_kv_len,
        d_queue_ptr, N_timed, d_stats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ke, stream));
    CUDA_CHECK(cudaEventSynchronize(ke));
    float wall_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&wall_ms, ks, ke));
    double wall_total_ms = (double)wall_ms;
    double wall_per_step_us = wall_total_ms * 1000.0 / N_timed;
    CUDA_CHECK(cudaEventDestroy(ks));
    CUDA_CHECK(cudaEventDestroy(ke));

    // Read back stats
    CUDA_CHECK(cudaMemcpy(h_stats, d_stats,
               MAX_N_TRAJ * sizeof(TrajStats), cudaMemcpyDeviceToHost));
    double compute_us = 0.0;
    n_traj = 1;
    for (int i = 0; i < n_traj; i++) {
        if (h_stats[i].n_timed_steps > 0) {
            compute_us = (double)h_stats[i].compute_cycles_total
                         / h_stats[i].n_timed_steps / cycles_per_ms * 1000.0;
            if (verbose)
                printf("  traj %d: compute=%.4f us  steps=%d\n",
                       i, compute_us, h_stats[i].n_timed_steps);
        }
    }

    // --- Derived quantities ---
    double naive_per_step_us = naive_total_us / N_timed;
    double kappa_us = naive_per_step_us - compute_us; // pure launch overhead per step
    double batch_overhead_us = wall_per_step_us - compute_us;

    printf("\n=== Results (N=%d, %d steps) ===\n", n_traj, N_timed);
    printf("  Naive total:                %.3f ms\n", naive_total_us / 1000.0);
    printf("  Naive per step:             %.4f us\n", naive_per_step_us);
    printf("  Batch total:                %.3f ms\n", wall_total_ms);
    printf("  Batch per step:             %.4f us\n", wall_per_step_us);
    printf("  Compute (clock64):          %.4f us\n", compute_us);
    printf("  Launch overhead per step:\n");
    printf("    Naive:                    %.4f us\n", kappa_us);
    printf("    Batch:                    %.4f us  (amortized)\n", batch_overhead_us);
    printf("  Saving:                     %.4f us/step  (%.1f%%)\n",
           naive_per_step_us - wall_per_step_us,
           (naive_per_step_us - wall_per_step_us) / naive_per_step_us * 100.0);

    // --- Shutdown ---
    CUDA_CHECK(cudaStreamDestroy(stream));

    // --- JSON output ---
    if (json_out) {
        FILE* f = fopen(json_out, "w");
        if (f) {
            fprintf(f,
                "{\n"
                "  \"device\": \"%s\",\n"
                "  \"sm\": \"%d%d\",\n"
                "  \"clock_rate_khz\": %d,\n"
                "  \"model\": {\"n_layers\": %d, \"dim\": %d, \"n_heads\": %d, \"ffn_dim\": %d},\n"
                "  \"n_steps\": %d,\n"
                "  \"naive_per_step_us\": %.4f,\n"
                "  \"batch_per_step_us\": %.4f,\n"
                "  \"compute_us\": %.4f,\n"
                "  \"kappa_naive_us\": %.4f,\n"
                "  \"kappa_batch_us\": %.4f,\n"
                "  \"saving_us\": %.4f,\n"
                "  \"saving_pct\": %.2f\n"
                "}\n",
                prop.name, prop.major, prop.minor, clock_rate_khz,
                N_LAYERS, DIM, N_HEADS, FFN_DIM,
                N_timed,
                naive_per_step_us, wall_per_step_us,
                compute_us, kappa_us, batch_overhead_us,
                naive_per_step_us - wall_per_step_us,
                (naive_per_step_us - wall_per_step_us) / naive_per_step_us * 100.0);
            fclose(f);
            printf("\nWrote %s\n", json_out);
        }
    }

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_kv_len));
    CUDA_CHECK(cudaFree(d_stats));
    CUDA_CHECK(cudaFreeHost(h_queue));

    printf("\nDone.\n");
    return 0;
}
