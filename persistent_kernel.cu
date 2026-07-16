// persistent_kernel.cu
//
// Persistent kernel with REAL transformer decode
// (LayerNorm -> MHA -> residual -> LayerNorm -> MLP -> residual) x N_LAYERS
// for agent-loop inference. The kernel stays resident across tool-call
// boundaries using a host-managed work queue.
//
// Two modes:
//   1. Persistent: launch once, queue-driven steps (no relaunch)
//   2. Naive:      fresh kernel launch every step (baseline)
//
// Build:
//   nvcc -O3 -arch=sm_89 persistent_kernel.cu -o persistent_kernel
//   (adjust -arch: sm_86 for RTX 30xx, sm_90 for H100, sm_89 for Ada Lovelace)

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
// Configuration (matches benchmark.py --real-model defaults)
// ---------------------------------------------------------------------------
#ifdef LARGE_MODEL
#define DIM          2048
#define N_HEADS      16
#define N_LAYERS     24
#else
#define DIM          256
#define N_HEADS      4
#define N_LAYERS     2
#endif

#define HEAD_DIM     (DIM / N_HEADS)   // 64 for small, 128 for large
#define MLP_RATIO    4
#define FFN_DIM      (DIM * MLP_RATIO) // 1024 for small, 8192 for large
#define MAX_SEQ_LEN  512
#define EPS          1e-5f

// ---------------------------------------------------------------------------
// Queue entry for agent-loop step
// ---------------------------------------------------------------------------
enum SlotState : int32_t {
    SLOT_EMPTY    = 0,
    SLOT_READY    = 1,
    SLOT_DONE     = 2,
    SLOT_SHUTDOWN = 3
};

struct QueueEntry {
    int32_t state;
    int32_t step_id;
    int32_t seq_len;            // current KV cache length before this step
    float   input[DIM];
    float   output[DIM];
};

// Device-side timing accumulators (clock64 cycles). Written by thread 0.
struct KernelStats {
    unsigned long long sigma_cycles_total;    // READY observed -> warps synced, pre-decode
    unsigned long long compute_cycles_total;  // decode_segment wall (clock64)
    int32_t            n_timed_steps;
    int32_t            _pad;
};

// Queue is allocated via cudaMalloc / mapped host memory and passed as a
// kernel parameter (not a __device__ symbol) for host/device coherence.

// ---------------------------------------------------------------------------
// Weight layout helpers (all weights packed per-layer for coalesced access)
// ---------------------------------------------------------------------------
// Per-layer offset table (in float units):
//   [0]  ln1_weight  DIM
//   [1]  ln1_bias    DIM
//   [2]  qkv_weight  DIM * 3*DIM
//   [3]  qkv_bias    3*DIM
//   [4]  attn_o_wt   DIM * DIM
//   [5]  attn_o_bias DIM
//   [6]  ln2_weight  DIM
//   [7]  ln2_bias    DIM
//   [8]  fc1_weight  DIM * FFN_DIM
//   [9]  fc1_bias    FFN_DIM
//   [10] fc2_weight  FFN_DIM * DIM
//   [11] fc2_bias    DIM

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

// KV cache layout: [layer][head][pos][dim]
// K_cache[layer][head * MAX_SEQ_LEN * HEAD_DIM + pos * HEAD_DIM + d]
#define KV_CACHE_LAYER_FLOATS (N_HEADS * MAX_SEQ_LEN * HEAD_DIM)

__device__ inline int kv_cache_len_offset(int layer) {
    return layer; // one int per layer
}

__device__ inline int kv_cache_offset(int layer, int head, int pos, int d) {
    return layer * KV_CACHE_LAYER_FLOATS + head * (MAX_SEQ_LEN * HEAD_DIM) + pos * HEAD_DIM + d;
}

// Global memory pointers passed to kernel
// We'll pack lengths in a separate int buffer
// Weights: [N_LAYERS * LAYER_FLOATS]
// KV cache: [N_LAYERS * N_HEADS * MAX_SEQ_LEN * HEAD_DIM]

// ---------------------------------------------------------------------------
// Host-side CUDA error checking
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
// Device helper: GELU activation (tanh approximation)
// ---------------------------------------------------------------------------
__device__ inline float gelu_fwd(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ---------------------------------------------------------------------------
// Device helper: GEMV  y[M] = W[M,N] @ x[N] + bias[M]
// Assumes all threads in block participate; use __syncthreads internally.
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
// Device helper: LayerNorm  x = (x - mean) / sqrt(var + eps) * w + b
// ---------------------------------------------------------------------------
__device__ void layer_norm(float* x, const float* w, const float* b,
                            int dim, float eps) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Sum
    float sum = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        sum += x[i];
    }

    // Reduction
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

    // Variance
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

    // Normalize
    for (int i = tid; i < dim; i += nthreads) {
        x[i] = (x[i] - mean) * inv_std * w[i] + (b ? b[i] : 0.0f);
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Device: Multi-head attention for ONE head (with KV cache update)
// Processes head_idx, assumes Q/K/V for this head are already split out.
// ---------------------------------------------------------------------------
__device__ void attention_head(
    const float* q_head,        // [HEAD_DIM]
    const float* k_head,        // [HEAD_DIM]  (new K to append)
    const float* v_head,        // [HEAD_DIM]  (new V to append)
    float* k_cache_head,        // [MAX_SEQ_LEN * HEAD_DIM]
    float* v_cache_head,        // [MAX_SEQ_LEN * HEAD_DIM]
    int old_seq_len,            // length BEFORE this step
    float* out_head             // [HEAD_DIM]  (output for this head)
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int new_len = old_seq_len + 1;

    // Step 1: Append K, V to cache
    int write_pos = old_seq_len;
    for (int d = tid; d < HEAD_DIM; d += nthreads) {
        k_cache_head[write_pos * HEAD_DIM + d] = k_head[d];
        v_cache_head[write_pos * HEAD_DIM + d] = v_head[d];
    }
    __syncthreads();

    // Step 2: Compute attention scores Q @ K_cache^T / sqrt(d)
    // scores[pos] = sum_d Q[d] * K_cache[pos][d]  for pos in 0..new_len-1
    // Shared memory for scores (max 512 elements)
    __shared__ float scores[MAX_SEQ_LEN];
    for (int pos = tid; pos < new_len; pos += nthreads) {
        float s = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            s += q_head[d] * k_cache_head[pos * HEAD_DIM + d];
        }
        scores[pos] = s * rsqrtf((float)HEAD_DIM);
    }
    __syncthreads();

    // Step 3: Stable softmax over scores[0:new_len-1]
    // Find max
    float my_max = -1e30f;
    for (int pos = tid; pos < new_len; pos += nthreads) {
        if (scores[pos] > my_max) my_max = scores[pos];
    }
    // Reduction for max
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

    // Exp and sum
    float my_sum = 0.0f;
    for (int pos = tid; pos < new_len; pos += nthreads) {
        scores[pos] = expf(scores[pos] - smax);
        my_sum += scores[pos];
    }
    __syncthreads();

    // Reduction for sum
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

    // Normalize scores
    float inv_sum = 1.0f / ssum;
    for (int pos = tid; pos < new_len; pos += nthreads) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Weighted sum of V_cache: out_head[d] = sum_pos scores[pos] * V_cache[pos][d]
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
// Shared-memory workspace for one decode step (block-wide, not per-thread).
// ~17 KB — fits comfortably in sm_89 shared memory.
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
// Device: Full decode segment for one agent step
// Runs the transformer forward pass, updates KV cache in-place.
// ---------------------------------------------------------------------------
__device__ void decode_segment(
    const float* input,   // [DIM] token embedding
    float* output,        // [DIM] output logits
    float* weights,       // [N_LAYERS * LAYER_FLOATS]
    float* k_cache,       // [N_LAYERS * N_HEADS * MAX_SEQ_LEN * HEAD_DIM]
    float* v_cache,       // [N_LAYERS * N_HEADS * MAX_SEQ_LEN * HEAD_DIM]
    int* kv_len,          // [N_LAYERS] current lengths, updated in-place
    DecodeWorkspace* ws   // block-shared working buffers
) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float* h = ws->h;

    // Copy input to h
    for (int i = tid; i < DIM; i += nthreads) {
        h[i] = input[i];
    }
    __syncthreads();

    for (int layer = 0; layer < N_LAYERS; layer++) {
        // --- Save residual ---
        for (int i = tid; i < DIM; i += nthreads) ws->residual[i] = h[i];
        __syncthreads();

        // --- Pre-norm LayerNorm ---
        for (int i = tid; i < DIM; i += nthreads) ws->normed[i] = h[i];
        __syncthreads();
        layer_norm(ws->normed,
                   weights + off_ln1_w(layer),
                   weights + off_ln1_b(layer),
                   DIM, EPS);

        // --- QKV projections ---
        gemv(ws->qkv, weights + off_qkv_w(layer), ws->normed,
             weights + off_qkv_b(layer), 3 * DIM, DIM);
        for (int i = tid; i < DIM; i += nthreads) {
            ws->q[i] = ws->qkv[i];
            ws->k[i] = ws->qkv[DIM + i];
            ws->v[i] = ws->qkv[2 * DIM + i];
        }
        __syncthreads();

        // --- Multi-head attention (process heads sequentially) ---
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

            for (int d = tid; d < HEAD_DIM; d += nthreads) {
                ws->attn_out[hd_off + d] = ws->out_head[d];
            }
            __syncthreads();
        }

        if (tid == 0) {
            kv_len[layer] = seq_len + 1;
        }
        __syncthreads();

        // --- Attention output projection + residual ---
        gemv(ws->attn_proj, weights + off_attn_o_w(layer), ws->attn_out,
             weights + off_attn_o_b(layer), DIM, DIM);
        for (int i = tid; i < DIM; i += nthreads) {
            h[i] = ws->residual[i] + ws->attn_proj[i];
        }
        __syncthreads();

        // --- MLP path ---
        for (int i = tid; i < DIM; i += nthreads) ws->residual[i] = h[i];
        __syncthreads();
        for (int i = tid; i < DIM; i += nthreads) ws->normed[i] = h[i];
        __syncthreads();
        layer_norm(ws->normed,
                   weights + off_ln2_w(layer),
                   weights + off_ln2_b(layer),
                   DIM, EPS);

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

    for (int i = tid; i < DIM; i += nthreads) {
        output[i] = h[i];
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Persistent kernel: launched ONCE, polls queue for work.
// Instrument sigma with clock64(): cycles from observing SLOT_READY until
// all warps are synced and about to enter decode_segment.
// ---------------------------------------------------------------------------
__global__ void persistent_agent_kernel(
    float* weights,
    float* k_cache,
    float* v_cache,
    int* kv_len,
    QueueEntry* queue,
    KernelStats* stats              // device ptr; may be null
) {
    if (blockIdx.x != 0) return;

    __shared__ int s_state;
    __shared__ unsigned long long s_t_ready;
    __shared__ DecodeWorkspace ws;

    if (threadIdx.x == 0)
        printf("persistent_kernel started, queue=%p\n", (void*)queue);

    // volatile: bypass L1 caching of host-mapped queue state writes
    volatile int* queue_state = &queue->state;

    while (true) {
        if (threadIdx.x == 0) {
            unsigned long long poll_iter = 0;
            while (true) {
                int s = *queue_state;
                if (s == SLOT_READY || s == SLOT_SHUTDOWN) {
                    s_state = s;
                    s_t_ready = clock64();  // first observation of READY/SHUTDOWN
                    break;
                }
                poll_iter++;
                if ((poll_iter & 0xFFFFFF) == 0) {
                    printf("  [kernel] polling... state=%d iter=%llu\n", s, poll_iter);
                }
                __threadfence();
            }
        }
        __syncthreads();  // broadcast s_state / s_t_ready to all warps

        int s = s_state;
        if (s == SLOT_SHUTDOWN) {
            if (threadIdx.x == 0) printf("  [kernel] SHUTDOWN received, exiting\n");
            return;
        }

        // All warps awake and synced: end of sigma window, start of compute.
        unsigned long long t_compute_start = clock64();

        decode_segment(
            queue->input,
            queue->output,
            weights,
            k_cache,
            v_cache,
            kv_len,
            &ws
        );

        unsigned long long t_compute_end = clock64();
        __syncthreads();

        if (threadIdx.x == 0) {
            if (stats) {
                stats->sigma_cycles_total   += (t_compute_start - s_t_ready);
                stats->compute_cycles_total += (t_compute_end - t_compute_start);
                stats->n_timed_steps        += 1;
            }
            __threadfence();
            queue->state = SLOT_DONE;
            __threadfence();
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Naive single-step kernel for baseline comparison
// Same forward pass, but exits after one step (fresh launch each time).
// ---------------------------------------------------------------------------
__global__ void transformer_step_kernel(
    const float* input, float* output, float* weights,
    float* k_cache, float* v_cache, int* kv_len
) {
    __shared__ DecodeWorkspace ws;
    decode_segment(input, output, weights, k_cache, v_cache, kv_len, &ws);
}

// ---------------------------------------------------------------------------
// Host: weight initialization
// ---------------------------------------------------------------------------
void init_weights(float* h_weights, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    int n_floats = N_LAYERS * LAYER_FLOATS;
    for (int i = 0; i < n_floats; i++) {
        h_weights[i] = dist(rng);
    }

    // Initialize LayerNorm weights to 1, biases to 0
    for (int layer = 0; layer < N_LAYERS; layer++) {
        float* ln1_w = h_weights + off_ln1_w(layer);
        float* ln1_b = h_weights + off_ln1_b(layer);
        float* ln2_w = h_weights + off_ln2_w(layer);
        float* ln2_b = h_weights + off_ln2_b(layer);
        for (int i = 0; i < DIM; i++) {
            ln1_w[i] = 1.0f;
            ln1_b[i] = 0.0f;
            ln2_w[i] = 1.0f;
            ln2_b[i] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Host: main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int T = 20;
    float tool_latency_ms = 20.0f;
    bool verbose = false;
    const char* json_out = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tool-latency-ms") == 0 && i + 1 < argc)
            tool_latency_ms = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            verbose = true;
        else if (strcmp(argv[i], "--json-out") == 0 && i + 1 < argc)
            json_out = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: persistent_kernel [--steps N] [--tool-latency-ms F] "
                   "[--json-out path] [--verbose]\n");
            return 0;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_rate_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0));
    // clockRate attribute is in kHz; clock64 ticks roughly once per SM cycle.
    double cycles_per_ms = (double)clock_rate_khz;

    printf("=== Persistent Kernel: Real Transformer Decode ===\n");
    printf("  Device: %s (sm_%d%d), clockRate=%d kHz\n",
           prop.name, prop.major, prop.minor, clock_rate_khz);
    printf("  Model: %d layers, dim=%d, %d heads, FFN=%d\n",
           N_LAYERS, DIM, N_HEADS, FFN_DIM);
    printf("  Steps: %d, Tool latency: %.1f ms\n", T, tool_latency_ms);
    printf("  DecodeWorkspace shared mem: %zu bytes\n", sizeof(DecodeWorkspace));
    printf("\n");

    int weights_floats = N_LAYERS * LAYER_FLOATS;
    int kv_floats_total = N_LAYERS * N_HEADS * MAX_SEQ_LEN * HEAD_DIM;

    std::vector<float> h_weights(weights_floats);
    std::vector<float> h_k_cache(kv_floats_total, 0.0f);
    std::vector<float> h_v_cache(kv_floats_total, 0.0f);
    std::vector<int>   h_kv_len(N_LAYERS, 0);
    init_weights(h_weights.data());

    float *d_weights, *d_k_cache, *d_v_cache;
    int *d_kv_len;
    KernelStats *d_stats;
    KernelStats h_stats = {};
    QueueEntry *h_entry, *h_readback;

    CUDA_CHECK(cudaMalloc(&d_weights, weights_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kv_len, N_LAYERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(KernelStats)));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(KernelStats)));

    QueueEntry* h_queue = nullptr;
    QueueEntry* d_queue = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_queue, sizeof(QueueEntry), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_queue, h_queue, 0));
    printf("  h_queue=%p d_queue=%p\n", (void*)h_queue, (void*)d_queue);
    CUDA_CHECK(cudaMallocHost(&h_entry, sizeof(QueueEntry)));
    CUDA_CHECK(cudaMallocHost(&h_readback, sizeof(QueueEntry)));

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weights_floats * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(), kv_floats_total * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(), kv_floats_total * sizeof(float),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kv_len, h_kv_len.data(), N_LAYERS * sizeof(int),
               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto wait_for_state = [&](int32_t expected, int timeout_ms) -> bool {
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
        do {
            if (std::chrono::steady_clock::now() > deadline) return false;
        } while (h_queue->state != expected);
        return true;
    };

    memset(h_queue, 0, sizeof(QueueEntry));
    h_queue->state = SLOT_EMPTY;
    cudaDeviceSynchronize();

    double sigma_ms = 0.0, compute_ms = 0.0, persistent_host_ms = 0.0;
    double naive_ms_per_step = 0.0, naive_kernel_per_step = 0.0;
    int N_steps = 50;

    // ── Persistent kernel mode ──────────────────────────────────────────
    {
        std::vector<int> zero_len(N_LAYERS, 0);
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(KernelStats)));

        cudaStream_t persistent_stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&persistent_stream, cudaStreamNonBlocking));
        persistent_agent_kernel<<<1, 256, 0, persistent_stream>>>(
            d_weights, d_k_cache, d_v_cache, d_kv_len, d_queue, d_stats);
        CUDA_CHECK(cudaGetLastError());

        for (int t = 0; t < T; t++) {
            auto step_start = std::chrono::high_resolution_clock::now();
            memset(h_queue, 0, sizeof(QueueEntry));
            h_queue->step_id = t;
            for (int i = 0; i < DIM; i++)
                h_queue->input[i] = 0.01f * (float)((t * DIM + i) % 100);
            std::atomic_signal_fence(std::memory_order_release);
            ((volatile int32_t*)&h_queue->state)[0] = SLOT_READY;

            if (!wait_for_state(SLOT_DONE, 30000)) {
                fprintf(stderr, "FATAL: persistent kernel not responding at step %d\n", t);
                CUDA_CHECK(cudaDeviceReset());
                return 1;
            }
            auto step_end = std::chrono::high_resolution_clock::now();
            if (verbose) {
                double ms = std::chrono::duration<double, std::milli>(
                                step_end - step_start).count();
                printf("  step %d: done in %.3f ms (persistent)\n", t, ms);
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds((int)tool_latency_ms));
        }

        // Timed run: reset stats so sigma excludes the tool-latency warm-up loop
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(KernelStats)));
        memset(h_queue, 0, sizeof(QueueEntry));
        ((volatile int32_t*)&h_queue->state)[0] = SLOT_EMPTY;

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < N_steps; t++) {
            memset(h_queue, 0, sizeof(QueueEntry));
            for (int i = 0; i < DIM; i++)
                h_queue->input[i] = 0.01f * (float)((t * DIM + i) % 100);
            std::atomic_signal_fence(std::memory_order_release);
            ((volatile int32_t*)&h_queue->state)[0] = SLOT_READY;
            if (!wait_for_state(SLOT_DONE, 30000)) {
                fprintf(stderr, "FATAL: persistent kernel not responding during timing\n");
                CUDA_CHECK(cudaDeviceReset());
                return 1;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double persistent_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        persistent_host_ms = persistent_ms / N_steps;

        CUDA_CHECK(cudaMemcpy(&h_stats, d_stats, sizeof(KernelStats),
                   cudaMemcpyDeviceToHost));
        if (h_stats.n_timed_steps > 0 && cycles_per_ms > 0.0) {
            sigma_ms = (double)h_stats.sigma_cycles_total
                       / (double)h_stats.n_timed_steps / cycles_per_ms;
            compute_ms = (double)h_stats.compute_cycles_total
                         / (double)h_stats.n_timed_steps / cycles_per_ms;
        }

        memset(h_queue, 0, sizeof(QueueEntry));
        ((volatile int32_t*)&h_queue->state)[0] = SLOT_SHUTDOWN;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CUDA_CHECK(cudaStreamDestroy(persistent_stream));

        printf("\n=== Persistent kernel (%d decode steps, no tool latency) ===\n",
               N_steps);
        printf("  Host wall total: %.3f ms\n", persistent_ms);
        printf("  Host wall per step: %.4f ms\n", persistent_host_ms);
        printf("  Device clock64 sigma  (READY->sync): %.6f ms  (%.1f cycles/step)\n",
               sigma_ms,
               h_stats.n_timed_steps
                   ? (double)h_stats.sigma_cycles_total / h_stats.n_timed_steps
                   : 0.0);
        printf("  Device clock64 compute (decode):     %.6f ms  (%.1f cycles/step)\n",
               compute_ms,
               h_stats.n_timed_steps
                   ? (double)h_stats.compute_cycles_total / h_stats.n_timed_steps
                   : 0.0);
        printf("  Timed steps (clock64): %d\n", h_stats.n_timed_steps);
    }

    // ── Naive baseline ──────────────────────────────────────────────────
    {
        std::vector<int> zero_len(N_LAYERS, 0);
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
                   cudaMemcpyHostToDevice));

        std::vector<float> h_input(DIM);
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, DIM * sizeof(float)));

        cudaEvent_t start_ev, end_ev;
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&end_ev));

        for (int i = 0; i < DIM; i++) h_input[i] = 0.01f;
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), DIM * sizeof(float),
                   cudaMemcpyHostToDevice));
        transformer_step_kernel<<<1, 256>>>(
            d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(), N_LAYERS * sizeof(int),
                   cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(start_ev, 0));
        for (int t = 0; t < N_steps; t++) {
            for (int i = 0; i < DIM; i++)
                h_input[i] = 0.01f * (float)((t * DIM + i) % 100);
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), DIM * sizeof(float),
                       cudaMemcpyHostToDevice));
            transformer_step_kernel<<<1, 256>>>(
                d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(end_ev, 0));
        CUDA_CHECK(cudaEventSynchronize(end_ev));

        float naive_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start_ev, end_ev));
        naive_ms_per_step = naive_ms / N_steps;

        // Kernel-only CUDA-event timing (no H2D per step).
        // This gives compute + launch overhead on the same wall-time clock
        // as the total, avoiding the clock64-vs-wall-time mismatch.
        cudaEvent_t ke_start, ke_end;
        CUDA_CHECK(cudaEventCreate(&ke_start));
        CUDA_CHECK(cudaEventCreate(&ke_end));
        CUDA_CHECK(cudaEventRecord(ke_start, 0));
        for (int t = 0; t < N_steps; t++) {
            transformer_step_kernel<<<1, 256>>>(
                d_input, d_output, d_weights, d_k_cache, d_v_cache, d_kv_len);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ke_end, 0));
        CUDA_CHECK(cudaEventSynchronize(ke_end));
        float naive_kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&naive_kernel_ms, ke_start, ke_end));
        naive_kernel_per_step = naive_kernel_ms / N_steps;

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(end_ev));
        CUDA_CHECK(cudaEventDestroy(ke_start));
        CUDA_CHECK(cudaEventDestroy(ke_end));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));

        printf("\n=== Naive (relaunch-per-step, %d decode steps) ===\n", N_steps);
        printf("  Total (H2D + launch + compute): %.3f ms\n", naive_ms);
        printf("  Per step (H2D + launch + compute): %.4f ms\n", naive_ms_per_step);
        printf("  Kernel-only (launch + compute, CUDA event): %.4f ms/step\n", naive_kernel_per_step);
    }

    double compute_ms_cuda_events = naive_kernel_per_step;
    double kappa_ms = naive_ms_per_step - compute_ms_cuda_events;
    if (kappa_ms < 0.0) kappa_ms = 0.0;

    printf("\n=== Derived overheads (real forward pass) ===\n");
    printf("  compute (CUDA event, naive kernel-only): %.6f ms\n", compute_ms_cuda_events);
    printf("  compute (clock64, persistent kernel):    %.6f ms (cross-check)\n", compute_ms);
    printf("  Total naive per step:                    %.6f ms\n", naive_ms_per_step);
    printf("  kappa (naive total - kernel-only cuda):  %.6f ms\n", kappa_ms);
    printf("  sigma (clock64 device-side):             %.6f ms\n", sigma_ms);
    printf("  kappa/sigma ratio:                       %.1fx\n",
           sigma_ms > 0.0 ? kappa_ms / sigma_ms : 0.0);

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
                "  \"sigma_ms\": %.8f,\n"
                "  \"sigma_source\": \"clock64_device_side_queue_poll\",\n"
                "  \"clock64_compute_ms\": %.8f,\n"
                "  \"clock64_compute_source\": \"clock64_decode_segment\",\n"
                "  \"cuda_event_compute_ms\": %.8f,\n"
                "  \"cuda_event_compute_source\": \"naive_kernel_only_cuda_event\",\n"
                "  \"naive_ms_per_step\": %.8f,\n"
                "  \"persistent_host_ms_per_step\": %.8f,\n"
                "  \"kappa_ms\": %.8f,\n"
                "  \"kappa_source\": \"naive_total_minus_kernel_only_cuda_event\",\n"
                "  \"sigma_cycles_total\": %llu,\n"
                "  \"compute_cycles_total\": %llu,\n"
                "  \"n_timed_steps\": %d\n"
                "}\n",
                prop.name, prop.major, prop.minor, clock_rate_khz,
                N_LAYERS, DIM, N_HEADS, FFN_DIM,
                N_steps,
                sigma_ms, compute_ms,
                compute_ms_cuda_events,
                naive_ms_per_step, persistent_host_ms,
                kappa_ms,
                (unsigned long long)h_stats.sigma_cycles_total,
                (unsigned long long)h_stats.compute_cycles_total,
                h_stats.n_timed_steps);
            fclose(f);
            printf("\nWrote %s\n", json_out);
        } else {
            fprintf(stderr, "WARNING: could not write %s\n", json_out);
        }
    }

    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_kv_len));
    CUDA_CHECK(cudaFree(d_stats));
    CUDA_CHECK(cudaFreeHost(h_queue));
    CUDA_CHECK(cudaFreeHost(h_entry));
    CUDA_CHECK(cudaFreeHost(h_readback));

    printf("\nDone.\n");
    return 0;
}

