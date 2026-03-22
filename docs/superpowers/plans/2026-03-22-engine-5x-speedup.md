# Engine 5x Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up echo-core inference engine from 789ms/token (~1.27 tok/s) to ~150ms/token (~6.5 tok/s)

**Architecture:** Fix matvec loop ordering for L1 cache efficiency + persistent thread pool for row-parallel execution across all matvec calls

**Tech Stack:** C++17, AVX2/FMA/F16C, std::thread, pthread

---

## 5x Justification

| Component | Current | After Fix | Gain |
|-----------|---------|-----------|------|
| Loop ordering | Input evicted from L1 248K times | Input stays in L1 (10KB < 48KB L1) | 1.5-2x |
| Threading | 1 core | 8-16 cores (memory BW limited) | 4-5x |
| Combined | 789ms/token | ~100-150ms/token | 5-8x |

Conservative: 789ms / 5 = 158ms = 6.3 tok/s (meets llama.cpp 6.5 tok/s target)

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `include/thread_pool.h` | CREATE | ThreadPool class with parallel_for |
| `src/kernels/matvec.cpp` | MODIFY | Loop ordering fix + threading |
| `include/kernels/matvec.h` | MODIFY | Signature updates, include thread_pool.h |
| `src/inference.cpp` | MODIFY | Create pool, pass to matvec |
| `include/inference.h` | MODIFY | ThreadPool member variable |
| `CMakeLists.txt` | MODIFY | Add pthread linking |

---

### Task 1: Create ThreadPool class

**Files:**
- Create: `include/thread_pool.h`

- [ ] **Step 1: Create include/thread_pool.h**

```cpp
#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <atomic>
#include <algorithm>

class ThreadPool {
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;

    std::function<void(uint32_t, uint32_t)> task_;
    uint32_t total_ = 0;
    uint32_t active_ = 0;
    uint32_t generation_ = 0;
    bool stop_ = false;

public:
    explicit ThreadPool(size_t n) {
        for (size_t i = 0; i < n; ++i)
            workers_.emplace_back([this, i] { worker(i); });
    }

    ~ThreadPool() {
        {
            std::lock_guard lk(mu_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t num_workers() const { return workers_.size(); }

    void parallel_for(uint32_t total, std::function<void(uint32_t, uint32_t)> fn) {
        if (total == 0) return;
        uint32_t nw = static_cast<uint32_t>(workers_.size());
        if (nw == 0) { fn(0, total); return; }

        // For small workloads, single-threaded is faster (no sync overhead)
        if (total < 128) { fn(0, total); return; }

        {
            std::lock_guard lk(mu_);
            task_ = std::move(fn);
            total_ = total;
            active_ = nw;
            ++generation_;
        }
        cv_start_.notify_all();

        // Main thread also works (chunk 0)
        uint32_t chunk = (total + nw) / (nw + 1);
        uint32_t main_end = std::min(chunk, total);
        task_(0, main_end);

        // Wait for workers
        std::unique_lock lk(mu_);
        cv_done_.wait(lk, [&] { return active_ == 0; });
    }

private:
    void worker(size_t idx) {
        uint32_t my_gen = 0;
        while (true) {
            std::unique_lock lk(mu_);
            cv_start_.wait(lk, [&] { return stop_ || generation_ > my_gen; });
            if (stop_) return;

            uint32_t nw = static_cast<uint32_t>(workers_.size());
            uint32_t chunk = (total_ + nw) / (nw + 1);
            uint32_t start = (idx + 1) * chunk;
            uint32_t end = std::min(start + chunk, total_);

            auto fn = task_;
            my_gen = generation_;
            lk.unlock();

            if (start < end) fn(start, end);

            lk.lock();
            if (--active_ == 0) cv_done_.notify_one();
        }
    }
};
```

- [ ] **Step 2: Verify it compiles**

Run: `g++ -std=c++17 -c -fsyntax-only -x c++ /dev/null -include include/thread_pool.h`

---

### Task 2: Fix matvec loop ordering

**Files:**
- Modify: `src/kernels/matvec.cpp`

- [ ] **Step 1: Rewrite matvec_fp16_fp32 with fixed loop ordering**

Replace the entire `matvec_fp16_fp32` function body (lines 18-55) with:

```cpp
template <uint32_t TILE_K, uint32_t TILE_M>
void matvec_fp16_fp32(const fp16_t* W, const float* x, float* y,
                       uint32_t M, uint32_t K) {
    // Fixed loop order: for m_tile → for m → for k_tile
    // This keeps the input vector x in L1 cache while processing a single row
    for (uint32_t m = 0; m < M; ++m) {
        const fp16_t* W_row = W + static_cast<size_t>(m) * K;

        __m256 acc_vec = _mm256_setzero_ps();
        float acc_scalar = 0.0f;

        uint32_t k = 0;
        for (; k + 8 <= K; k += 8) {
            __m128i w16 = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(W_row + k));
            __m256 w32 = _mm256_cvtph_ps(w16);
            __m256 xv  = _mm256_loadu_ps(x + k);
            acc_vec = _mm256_fmadd_ps(w32, xv, acc_vec);
        }

        // Scalar tail
        for (; k < K; ++k) {
            acc_scalar += fp16_to_fp32(W_row[k]) * x[k];
        }

        y[m] += hsum256_ps(acc_vec) + acc_scalar;
    }
}
```

**Key changes:**
- Removed tiling (was causing cache thrashing, not helping with K=2560)
- Single row processed completely before next row
- Input vector x stays in L1 (K=2560 × 4 bytes = 10KB < 48KB L1)
- AVX2 accumulator + scalar tail, same math, different order

- [ ] **Step 2: Update explicit instantiations**

Replace lines 57-59 with:
```cpp
template void matvec_fp16_fp32<1024, 512>(const fp16_t*, const float*, float*, uint32_t, uint32_t);
template void matvec_fp16_fp32<2048, 1024>(const fp16_t*, const float*, float*, uint32_t, uint32_t);
```

(These stay the same, but the TILE_K/TILE_M params are now unused in the body - that's fine, we keep the signature for API compatibility.)

- [ ] **Step 3: Update matvec_dispatch**

Replace lines 64-68 with:
```cpp
void matvec_dispatch(const fp16_t* W, const float* x, float* y,
                     uint32_t M, uint32_t K, const ModelConfig& config) {
    (void)config;
    matvec_fp16_fp32<Intel13500H_Tiles::TILE_K, Intel13500H_Tiles::TILE_M>(W, x, y, M, K);
}
```

(No change needed yet - threading comes in Task 3)

- [ ] **Step 4: Build and run existing tests**

Run: `cmake --build build -j$(nproc) && ./build/inference_test`
Expected: PASS (numerical results should be correct, possibly slightly different FP order)

---

### Task 3: Add threading to matvec

**Files:**
- Modify: `src/kernels/matvec.cpp` (add threaded dispatch)
- Modify: `include/kernels/matvec.h` (update signatures)
- Modify: `include/thread_pool.h` (already created in Task 1)

- [ ] **Step 1: Update include/kernels/matvec.h**

Add include at top:
```cpp
#include "thread_pool.h"
```

Update signatures:
```cpp
// Tiled AVX2 matrix-vector multiply with optional threading
template <uint32_t TILE_K, uint32_t TILE_M>
void matvec_fp16_fp32(const fp16_t* W, const float* x, float* y,
                       uint32_t M, uint32_t K, ThreadPool* pool = nullptr);

// Dispatch with optional thread pool
void matvec_dispatch(const fp16_t* W, const float* x, float* y,
                     uint32_t M, uint32_t K, const ModelConfig& config,
                     ThreadPool* pool = nullptr);
```

- [ ] **Step 2: Update matvec_fp16_fp32 in matvec.cpp with threading**

Replace the function body with threaded version:

```cpp
template <uint32_t TILE_K, uint32_t TILE_M>
void matvec_fp16_fp32(const fp16_t* W, const float* x, float* y,
                       uint32_t M, uint32_t K, ThreadPool* pool) {
    auto process_rows = [&](uint32_t m_start, uint32_t m_end) {
        for (uint32_t m = m_start; m < m_end; ++m) {
            const fp16_t* W_row = W + static_cast<size_t>(m) * K;

            __m256 acc_vec = _mm256_setzero_ps();
            float acc_scalar = 0.0f;

            uint32_t k = 0;
            for (; k + 8 <= K; k += 8) {
                __m128i w16 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(W_row + k));
                __m256 w32 = _mm256_cvtph_ps(w16);
                __m256 xv  = _mm256_loadu_ps(x + k);
                acc_vec = _mm256_fmadd_ps(w32, xv, acc_vec);
            }

            for (; k < K; ++k) {
                acc_scalar += fp16_to_fp32(W_row[k]) * x[k];
            }

            y[m] += hsum256_ps(acc_vec) + acc_scalar;
        }
    };

    if (pool && M >= 128) {
        pool->parallel_for(M, process_rows);
    } else {
        process_rows(0, M);
    }
}
```

- [ ] **Step 3: Update matvec_dispatch to pass pool**

```cpp
void matvec_dispatch(const fp16_t* W, const float* x, float* y,
                     uint32_t M, uint32_t K, const ModelConfig& config,
                     ThreadPool* pool) {
    (void)config;
    matvec_fp16_fp32<Intel13500H_Tiles::TILE_K, Intel13500H_Tiles::TILE_M>(
        W, x, y, M, K, pool);
}
```

- [ ] **Step 4: Update explicit instantiations**

```cpp
template void matvec_fp16_fp32<1024, 512>(const fp16_t*, const float*, float*,
                                           uint32_t, uint32_t, ThreadPool*);
template void matvec_fp16_fp32<2048, 1024>(const fp16_t*, const float*, float*,
                                            uint32_t, uint32_t, ThreadPool*);
```

- [ ] **Step 5: Build and test**

Run: `cmake --build build -j$(nproc) && ./build/inference_test`
Expected: PASS

---

### Task 4: Integrate with InferenceEngine

**Files:**
- Modify: `include/inference.h`
- Modify: `src/inference.cpp`

- [ ] **Step 1: Add ThreadPool member to inference.h**

Add include:
```cpp
#include "thread_pool.h"
```

Add member variable (after kv_cache_ declaration, around line 18):
```cpp
    std::unique_ptr<ThreadPool> pool_;     // thread pool for parallel matvec
```

- [ ] **Step 2: Create ThreadPool in InferenceEngine constructor**

In `src/inference.cpp`, constructor body (around line 45, after kv_cache_ creation):
```cpp
    uint32_t n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 8;
    pool_ = std::make_unique<ThreadPool>(n_threads);
```

Add include at top of inference.cpp if not already there:
```cpp
#include <thread>
```

- [ ] **Step 3: Pass pool to matvec in inference.cpp**

Update the `matvec` method (line 523-527):
```cpp
void InferenceEngine::matvec(const float* input, float* output,
                              const fp16_t* weight, uint32_t rows, uint32_t cols) {
    matvec_dispatch(weight, input, output, rows, cols, config_, pool_.get());
}
```

- [ ] **Step 4: Build and test**

Run: `cmake --build build -j$(nproc) && ./build/inference_test`
Expected: PASS

---

### Task 5: Update CMakeLists.txt and final verification

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add pthread linking**

Add to both echo_core and inference_test targets:
```cmake
find_package(Threads REQUIRED)
```

Add after `target_link_libraries` for echo_core (or add new line):
```cmake
target_link_libraries(echo_core PRIVATE Threads::Threads)
```

And for inference_test:
```cmake
target_link_libraries(inference_test PRIVATE Threads::Threads m)
```

- [ ] **Step 2: Full rebuild and test**

```bash
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
./inference_test
./quant_test
```

- [ ] **Step 3: Run with real model and benchmark**

```bash
./echo_core /path/to/model.gguf
```

Measure tok/s and compare with baseline (1.27 tok/s → target 6.5 tok/s).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "perf: 5-8x engine speedup via matvec loop fix + threading

- Fix matvec loop ordering (m→k instead of k→m) for L1 cache efficiency
- Add persistent ThreadPool with parallel_for
- Thread all matvec calls when M >= 128
- Remove unnecessary tiling (K=2560 fits in L1)"
```

---

## Performance Measurement Points

After each task, measure with:
```bash
./echo_core model.gguf
# Interactive REPL outputs: "[N tokens, X ms, Y ms/tok]"
# tok/s = 1000 / ms/tok
```

| Stage | Expected ms/token | Expected tok/s |
|-------|-------------------|----------------|
| Baseline | 789 | 1.27 |
| After Task 2 (loop fix) | ~350-500 | ~2-3 |
| After Task 3+4 (threading) | ~100-150 | ~6.5-10 |
