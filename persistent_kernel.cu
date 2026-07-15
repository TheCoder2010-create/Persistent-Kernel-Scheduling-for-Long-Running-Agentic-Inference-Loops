// persistent_kernel.cu
//
// Reference scaffold for a persistent (megakernel-style) GPU kernel that
// stays resident across agent-loop steps instead of being relaunched at
// every model call. Pairs with paper.tex, Section 4 ("Reference
// Implementation") and Algorithm 1.
//
// This is a SCAFFOLD: the decode_segment() stub is where a real forward
// pass (e.g. a Mirage-compiled task graph, or a hand-written attention +
// MLP block) should be plugged in. The queueing / residency / yield
// mechanics around it are the actual contribution of this file.
//
// Build:
//   nvcc -O3 -arch=sm_80 persistent_kernel.cu -o persistent_kernel
// (adjust -arch for your GPU: sm_86 for RTX 30xx/40xx-class, sm_89 for
// RTX 4050/4060/4070-class Ada Lovelace parts, sm_90 for H100.)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------
// Device-visible work queue. Host writes QueueEntry structs; the
// persistent kernel polls `state` on each entry instead of the kernel
// itself being torn down and relaunched between agent steps.
// ---------------------------------------------------------------------

enum SlotState : int32_t {
    SLOT_EMPTY     = 0,  // no work yet — device should keep polling
    SLOT_READY     = 1,  // host has pushed a new step's input
    SLOT_DONE      = 2,  // device has written its output, host may read
    SLOT_SHUTDOWN  = 3   // host requests kernel exit
};

struct QueueEntry {
    int32_t state;        // SlotState, written by host (READY) / device (DONE)
    int32_t step_id;      // which agent-loop step this corresponds to
    int32_t input_len;    // length of encoded input in `input` buffer
    int32_t output_len;   // length written to `output` buffer by device
    float   input[4096];  // toy fixed-size buffer standing in for encoded
                           // agent state (token embeddings / KV deltas).
                           // Replace with your real per-step payload.
    float   output[4096];
};

// Single queue slot is enough for a strictly sequential agent loop
// (one in-flight step at a time). For concurrent multi-agent sessions,
// extend this to N slots, one per session, and have the kernel's warp
// groups each own a slot (this is the multi-agent batching path
// discussed in paper.tex Section 4, point 3).
__device__ QueueEntry g_queue;

// ---------------------------------------------------------------------
// Decode-segment stub. Replace this with the real fused forward pass.
// Kept trivial here so the file compiles and runs standalone to
// demonstrate the residency/queueing mechanism itself.
// ---------------------------------------------------------------------
__device__ void decode_segment(QueueEntry* entry) {
    // Toy "compute": normalize-ish pass over the input buffer.
    // A real implementation fuses attention + MLP + sampling here,
    // following the SM-level task-graph approach of MPK-style
    // megakernels (see paper.tex, Section 3).
    int len = entry->input_len;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        entry->output[i] = entry->input[i] * 1.0f; // placeholder op
    }
    entry->output_len = len;
}

// ---------------------------------------------------------------------
// The persistent kernel itself: launched ONCE for the whole agent
// session. It polls g_queue.state instead of exiting between steps.
// ---------------------------------------------------------------------
__global__ void persistent_agent_kernel(unsigned long long poll_window_cycles) {
    if (blockIdx.x != 0) return; // single-block resident kernel for clarity

    while (true) {
        unsigned long long start = clock64();
        bool got_work = false;

        // Poll for new work within the yield window before cooperatively
        // releasing (in a real system this would signal a scheduler,
        // e.g. via cudaStreamAttachMemAsync-style cooperation or MPS;
        // here we just busy-poll with a bounded spin as a stand-in).
        while ((clock64() - start) < (long long)poll_window_cycles) {
            int32_t s = g_queue.state;
            if (s == SLOT_READY) { got_work = true; break; }
            if (s == SLOT_SHUTDOWN) return;
        }

        if (!got_work) {
            // Cooperative yield point: in a production design this is
            // where SMs would be released back to the scheduler so
            // other agent sessions / workloads can use them while this
            // session's tool call is still in flight (Algorithm 1,
            // line 6). Here we just continue polling.
            continue;
        }

        __syncthreads();
        decode_segment(&g_queue);
        __syncthreads();

        if (threadIdx.x == 0) {
            g_queue.state = SLOT_DONE;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------
// Host-side driver: this is the part that changes vs. a naive
// relaunch-per-step design. The kernel is launched ONCE; each agent
// step becomes a queue push/pop instead of a cudaLaunchKernel call.
// ---------------------------------------------------------------------
int main() {
    // Two streams: kernel_stream holds the persistent kernel (it never
    // completes, so nothing else can go on it). memcpy_stream is used for
    // all host<->device transfers so they are not serialised behind the
    // running kernel.
    cudaStream_t kernel_stream, memcpy_stream;
    cudaStreamCreate(&kernel_stream);
    // Non-blocking so memcpy_stream does not synchronise with the default
    // stream and is not blocked by the persistent kernel on kernel_stream.
    cudaStreamCreateWithFlags(&memcpy_stream, cudaStreamNonBlocking);

    // Zero-initialise the device queue so the kernel sees SLOT_EMPTY (0)
    // from the very start.
    {
        QueueEntry zero{};
        cudaMemcpyToSymbol(g_queue, &zero, sizeof(QueueEntry), 0,
                           cudaMemcpyHostToDevice);
    }

    // Launch the resident kernel once. Poll window in clock cycles;
    // tune to trade CPU spin cost vs. responsiveness to new steps.
    const unsigned long long poll_window_cycles = 100000ULL;
    persistent_agent_kernel<<<1, 256, 0, kernel_stream>>>(poll_window_cycles);

    // Use page-locked (pinned) host memory for async copies.
    QueueEntry* host_entry = nullptr;
    cudaMallocHost(&host_entry, sizeof(QueueEntry));
    memset(host_entry, 0, sizeof(QueueEntry));

    QueueEntry* readback = nullptr;
    cudaMallocHost(&readback, sizeof(QueueEntry));

    const int T = 20; // number of agent-loop steps to simulate

    for (int t = 0; t < T; ++t) {
        auto step_start = std::chrono::high_resolution_clock::now();

        // 1) Host encodes current agent state (stand-in for real
        //    tokenization / embedding of the latest tool result).
        host_entry->step_id  = t;
        host_entry->input_len = 128;
        for (int i = 0; i < 128; ++i)
            host_entry->input[i] = static_cast<float>(t + i);
        // Write state last so the kernel does not see SLOT_READY before
        // the payload is in place.
        host_entry->state = SLOT_READY;

        cudaMemcpyToSymbolAsync(g_queue, host_entry, sizeof(QueueEntry),
                                0, cudaMemcpyHostToDevice, memcpy_stream);
        cudaStreamSynchronize(memcpy_stream); // ensure write is visible

        // 2) Wait for device to flip state to SLOT_DONE. No kernel
        //    relaunch happens here — this replaces cudaLaunchKernel().
        do {
            cudaMemcpyFromSymbolAsync(readback, g_queue, sizeof(QueueEntry),
                                     0, cudaMemcpyDeviceToHost, memcpy_stream);
            cudaStreamSynchronize(memcpy_stream);
        } while (readback->state != SLOT_DONE);

        auto step_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(
                        step_end - step_start).count();
        printf("step %d: decode segment done in %.3f ms (no kernel relaunch)\n",
               t, ms);

        // 3) Simulate an external tool call (opaque latency `ell_t` in
        //    the paper's cost model). Replace with a real async tool
        //    invocation (HTTP call, DB query, subprocess, etc).
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    // Shut down the resident kernel cleanly.
    host_entry->state = SLOT_SHUTDOWN;
    cudaMemcpyToSymbolAsync(g_queue, host_entry, sizeof(QueueEntry),
                            0, cudaMemcpyHostToDevice, memcpy_stream);
    cudaStreamSynchronize(memcpy_stream);

    // Give the kernel a moment to observe SHUTDOWN and return.
    cudaStreamSynchronize(kernel_stream);

    cudaFreeHost(host_entry);
    cudaFreeHost(readback);
    cudaStreamDestroy(memcpy_stream);
    cudaStreamDestroy(kernel_stream);

    printf("Resident kernel exited after %d agent steps, single launch.\n", T);
    return 0;
}
