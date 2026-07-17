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
    float* k_cache,           // [MAX_N_TRAJ * N_LAYERS * KV_CACHE_LAYER_FLOATS]
    float* v_cache,           // [MAX_N_TRAJ * N_LAYERS * KV_CACHE_LAYER_FLOATS]
    int*   kv_len,            // [MAX_N_TRAJ * N_LAYERS]
    QueueEntry* queue,        // [MAX_N_TRAJ] mapped host memory
    int    n_traj,
    TrajStats* stats          // [MAX_N_TRAJ] device; may be null
) {
    if (blockIdx.x != 0) return;

    __shared__ DecodeWorkspace ws;

    // Per-slot communication (set by leader thread, broadcast via __syncthreads)
    __shared__ int      s_slot_state;
    __shared__ int      s_slot_traj_id;
    __shared__ int      s_slot_idx;
    __shared__ unsigned long long s_t_ready;

    // We need indirect access: for a given queue slot, read its volatile state.
    // Each slot has its own state field.  We iterate i=0..n_traj-1 and check
    // queue[i].state.  Since queue is mapped host memory, reads are uncached
    // and we must use volatile to prevent compiler hoisting.
    volatile int* slot_state_base = &queue[0].state;

    while (true) {
        bool any_work = false;

        for (int i = 0; i < n_traj; i++) {
            if (threadIdx.x == 0) {
                int s = *(volatile int*)(slot_state_base + i * sizeof(QueueEntry) / sizeof(int));

                if (s == SLOT_SHUTDOWN) {
                    // Don't exit yet — other slots may still have work.
                    // Flag this slot as seen-shutdown locally.
                    s_slot_state = SLOT_SHUTDOWN;
                    s_slot_idx = i;
                    any_work = true;  // don't nanosleep — keep polling others
                } else if (s == SLOT_READY) {
                    s_slot_state = SLOT_READY;
                    s_slot_idx = i;
                    s_slot_traj_id = queue[i].trajectory_id;
                    s_t_ready = clock64();
                    any_work = true;
                }
            }
            __syncthreads();

            int state = s_slot_state;
            if (state == SLOT_READY) {
                int idx = s_slot_idx;
                int tid = s_slot_traj_id;

                // End of sigma window: all warps are awake and synced
                unsigned long long t_compute_start = clock64();

                // Decode this trajectory's segment
                decode_segment(
                    queue[idx].input,
                    queue[idx].output,
                    weights,
                    k_cache + tid * N_LAYERS * KV_CACHE_LAYER_FLOATS,
                    v_cache + tid * N_LAYERS * KV_CACHE_LAYER_FLOATS,
                    kv_len + tid * N_LAYERS,
                    &ws
                );

                unsigned long long t_compute_end = clock64();
                __syncthreads();

                if (threadIdx.x == 0) {
                    // Record stats
                    if (stats) {
                        stats[tid].sigma_cycles_total   += (t_compute_start - s_t_ready);
                        stats[tid].compute_cycles_total += (t_compute_end - t_compute_start);
                        stats[tid].n_timed_steps        += 1;
                    }
                    // Signal done
                    __threadfence();
                    *(volatile int*)(slot_state_base + idx * sizeof(QueueEntry) / sizeof(int)) = SLOT_DONE;
                    __threadfence();
                }
                __syncthreads();
            }
        }

        // Check if ALL slots are SHUTDOWN -> exit
        if (threadIdx.x == 0) {
            bool all_shutdown = true;
            for (int i = 0; i < n_traj; i++) {
                int s = *(volatile int*)(slot_state_base + i * sizeof(QueueEntry) / sizeof(int));
                if (s != SLOT_SHUTDOWN) {
                    all_shutdown = false;
                    break;
                }
            }
            if (all_shutdown) {
                s_slot_state = SLOT_SHUTDOWN;
            } else {
                s_slot_state = SLOT_EMPTY;  // keep going
            }
        }
        __syncthreads();
        if (s_slot_state == SLOT_SHUTDOWN) {
            if (threadIdx.x == 0)
                printf("  [kernel] all %d slots SHUTDOWN, exiting\n", n_traj);
            return;
        }

        // Yield briefly if no work found this cycle
        if (!any_work) {
            __nanosleep(200);
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

    std::vector<float> h_weights(weights_floats);
    std::vector<float> h_k_cache(kv_floats_total, 0.0f);
    std::vector<float> h_v_cache(kv_floats_total, 0.0f);
    std::vector<int>   h_kv_len(MAX_N_TRAJ * N_LAYERS, 0);
    init_weights(h_weights.data());

    float *d_weights, *d_k_cache, *d_v_cache;
    int *d_kv_len;
    TrajStats *d_stats, h_stats[MAX_N_TRAJ] = {};

    CUDA_CHECK(cudaMalloc(&d_weights, weights_floats * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_cache, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_cache, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kv_len, MAX_N_TRAJ * N_LAYERS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_stats, MAX_N_TRAJ * sizeof(TrajStats)));
    CUDA_CHECK(cudaMemset(d_stats, 0, MAX_N_TRAJ * sizeof(TrajStats)));

    // Queue: mapped host memory array of MAX_N_TRAJ entries
    QueueEntry* h_queue = nullptr;
    QueueEntry* d_queue_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_queue, MAX_N_TRAJ * sizeof(QueueEntry),
                              cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_queue_ptr, h_queue, 0));

    // Initialize queue
    memset(h_queue, 0, MAX_N_TRAJ * sizeof(QueueEntry));
    for (int i = 0; i < MAX_N_TRAJ; i++) {
        h_queue[i].state = SLOT_EMPTY;
        h_queue[i].trajectory_id = i;
    }

    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(),
               weights_floats * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k_cache, h_k_cache.data(),
               kv_floats_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_cache, h_v_cache.data(),
               kv_floats_total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kv_len, h_kv_len.data(),
               MAX_N_TRAJ * N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto wait_for_slot = [&](int slot, int32_t expected,
                              int timeout_ms = 30000) -> bool {
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
        volatile int* s = &h_queue[slot].state;
        do {
            int v = *s;
            if (v == expected) return true;
            if (std::chrono::steady_clock::now() > deadline) return false;
        } while (true);
    };

    auto wait_for_all_done = [&](int timeout_ms = 30000) -> bool {
        auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(timeout_ms);
        do {
            bool all_done = true;
            for (int i = 0; i < n_traj; i++) {
                if (h_queue[i].state != SLOT_DONE) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) return true;
            if (std::chrono::steady_clock::now() > deadline) return false;
        } while (true);
    };

    // Reset all slots before launch
    for (int i = 0; i < MAX_N_TRAJ; i++) {
        memset(&h_queue[i], 0, sizeof(QueueEntry));
        h_queue[i].state = SLOT_EMPTY;
        h_queue[i].trajectory_id = i;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Launch persistent kernel ---
    cudaStream_t persistent_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&persistent_stream,
                                          cudaStreamNonBlocking));
    persistent_multi_agent_kernel<<<1, 256, 0, persistent_stream>>>(
        d_weights, d_k_cache, d_v_cache, d_kv_len,
        d_queue_ptr, n_traj, d_stats);
    CUDA_CHECK(cudaGetLastError());
    printf("  Persistent kernel launched (%d trajectories)\n", n_traj);

    // --- Warmup / tool-latency loop ---
    for (int t = 0; t < T; t++) {
        // Submit all N trajectories
        for (int i = 0; i < n_traj; i++) {
            memset(&h_queue[i], 0, sizeof(QueueEntry));
            h_queue[i].state = SLOT_EMPTY;
            h_queue[i].trajectory_id = i;
            for (int d = 0; d < DIM; d++)
                h_queue[i].input[d] = 0.01f * (float)((t * DIM + d) % 100);
            std::atomic_signal_fence(std::memory_order_release);
            h_queue[i].state = SLOT_READY;
        }

        // Wait for all N to complete
        if (!wait_for_all_done(60000)) {
            fprintf(stderr, "FATAL: kernel not responding at step %d\n", t);
            CUDA_CHECK(cudaDeviceReset());
            return 1;
        }

        if (verbose) {
            printf("  step %d: all %d trajectories done\n", t, n_traj);
        }

        std::this_thread::sleep_for(
            std::chrono::milliseconds((int)tool_latency_ms));
    }

    // --- Timed run ---
    // Reset KV caches
    std::vector<int> zero_kv(MAX_N_TRAJ * N_LAYERS, 0);
    CUDA_CHECK(cudaMemcpy(d_kv_len, zero_kv.data(),
               MAX_N_TRAJ * N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_k_cache, 0, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, kv_floats_total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_stats, 0, MAX_N_TRAJ * sizeof(TrajStats)));

    // Reset queue
    for (int i = 0; i < MAX_N_TRAJ; i++) {
        memset(&h_queue[i], 0, sizeof(QueueEntry));
        h_queue[i].state = SLOT_EMPTY;
        h_queue[i].trajectory_id = i;
    }

    int N_timed = 50;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < N_timed; t++) {
        for (int i = 0; i < n_traj; i++) {
            memset(&h_queue[i], 0, sizeof(QueueEntry));
            h_queue[i].trajectory_id = i;
            for (int d = 0; d < DIM; d++)
                h_queue[i].input[d] = 0.01f * (float)((t * DIM + d) % 100);
            std::atomic_signal_fence(std::memory_order_release);
            h_queue[i].state = SLOT_READY;
        }
        if (!wait_for_all_done(60000)) {
            fprintf(stderr, "FATAL during timed run\n");
            CUDA_CHECK(cudaDeviceReset());
            return 1;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double wall_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double wall_per_step = wall_total_ms / N_timed;

    // Read back stats
    CUDA_CHECK(cudaMemcpy(h_stats, d_stats,
               MAX_N_TRAJ * sizeof(TrajStats), cudaMemcpyDeviceToHost));

    // Compute per-trajectory averages
    double avg_sigma_us = 0.0, avg_compute_us = 0.0;
    int total_timed_steps = 0;
    for (int i = 0; i < n_traj; i++) {
        if (h_stats[i].n_timed_steps > 0) {
            double s = (double)h_stats[i].sigma_cycles_total
                       / h_stats[i].n_timed_steps / cycles_per_ms * 1000.0;
            double c = (double)h_stats[i].compute_cycles_total
                       / h_stats[i].n_timed_steps / cycles_per_ms * 1000.0;
            avg_sigma_us += s;
            avg_compute_us += c;
            total_timed_steps += h_stats[i].n_timed_steps;
            if (verbose) {
                printf("  traj %d: sigma=%.4f us  compute=%.4f us  steps=%d\n",
                       i, s, c, h_stats[i].n_timed_steps);
            }
        }
    }
    avg_sigma_us /= n_traj;
    avg_compute_us /= n_traj;
    double sigma_us = avg_sigma_us;
    double compute_us = avg_compute_us;

    // --- Naive baseline (single trajectory, N repeats) ---
    double naive_kernel_only_us = 0.0;
    {
        std::vector<int> zero_len(N_LAYERS, 0);
        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(),
                   N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));

        std::vector<float> h_input(DIM);
        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, DIM * sizeof(float)));

        cudaEvent_t ks, ke;
        CUDA_CHECK(cudaEventCreate(&ks));
        CUDA_CHECK(cudaEventCreate(&ke));

        // Use one trajectory's KV cache region for naive baseline
        float* naive_k = d_k_cache;  // trajectory 0
        float* naive_v = d_v_cache;
        int* naive_len = d_kv_len;

        CUDA_CHECK(cudaMemcpy(d_kv_len, zero_len.data(),
                   N_LAYERS * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(ks, 0));
        for (int t = 0; t < N_timed; t++) {
            h_input[0] = 0.01f * (float)(t % 100);
            CUDA_CHECK(cudaMemcpy(d_input, h_input.data(),
                       DIM * sizeof(float), cudaMemcpyHostToDevice));
            transformer_step_kernel<<<1, 256>>>(
                d_input, d_output, d_weights, naive_k, naive_v, naive_len);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ke, 0));
        CUDA_CHECK(cudaEventSynchronize(ke));
        float naive_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ks, ke));
        naive_kernel_only_us = naive_ms / N_timed * 1000.0;

        CUDA_CHECK(cudaEventDestroy(ks));
        CUDA_CHECK(cudaEventDestroy(ke));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    // --- Compute derived quantities ---
    // kappa = naive kernel-only per step
    double kappa_us = naive_kernel_only_us;

    // Multi-trajectory analysis
    double per_traj_per_step_persistent_us = (wall_per_step * 1000.0) / n_traj;
    double per_traj_per_step_naive_us = kappa_us + compute_us;
    double delta_us_per_traj_step = per_traj_per_step_naive_us - per_traj_per_step_persistent_us;
    double delta_us_total = (kappa_us * n_traj) - (kappa_us + sigma_us * n_traj);

    printf("\n=== Results (N=%d, %d steps) ===\n", n_traj, N_timed);
    printf("  Wall total:               %.3f ms\n", wall_total_ms);
    printf("  Wall per step (all traj): %.4f ms\n", wall_per_step);
    printf("  Wall per traj per step:   %.4f us\n", per_traj_per_step_persistent_us);
    printf("\n  sigma (clock64 avg):      %.6f us\n", sigma_us);
    printf("  compute (clock64 avg):    %.6f us\n", compute_us);
    printf("  kappa (naive kernel-only):%.6f us\n", kappa_us);
    printf("\n  Delta per traj step:      %.4f us\n", delta_us_per_traj_step);
    printf("  Delta total (vs Nxnaive): %.4f us\n", delta_us_total);
    printf("  %saving:                   %.4f%%\n",
           delta_us_total > 0 ? "S" : "S",
           delta_us_total / (kappa_us * n_traj + compute_us * n_traj) * 100.0);

    // --- Shutdown ---
    for (int i = 0; i < n_traj; i++) {
        h_queue[i].state = SLOT_SHUTDOWN;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CUDA_CHECK(cudaStreamDestroy(persistent_stream));

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
                "  \"n_trajectories\": %d,\n"
                "  \"n_steps\": %d,\n"
                "  \"wall_total_ms\": %.6f,\n"
                "  \"wall_per_step_ms\": %.6f,\n"
                "  \"sigma_us\": %.8f,\n"
                "  \"compute_us\": %.8f,\n"
                "  \"kappa_us\": %.8f,\n"
                "  \"delta_us_per_traj_step\": %.8f,\n"
                "  \"delta_us_total\": %.8f,\n"
                "  \"saving_pct\": %.4f\n"
                "}\n",
                prop.name, prop.major, prop.minor, clock_rate_khz,
                N_LAYERS, DIM, N_HEADS, FFN_DIM,
                n_traj, N_timed,
                wall_total_ms, wall_per_step,
                sigma_us, compute_us, kappa_us,
                delta_us_per_traj_step,
                delta_us_total,
                delta_us_total / (kappa_us * n_traj + compute_us * n_traj) * 100.0);
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
