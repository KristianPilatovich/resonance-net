#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <string>

#ifdef RNET_NCCL
#include <nccl.h>
#endif

namespace rnet {

struct DistState {
    int world_size = 1;
    int rank = 0;
    int local_rank = 0;
#ifdef RNET_NCCL
    ncclComm_t comm = nullptr;
#endif
    bool active() const { return world_size > 1; }
    bool is_master() const { return rank == 0; }
};

// Reads RANK, WORLD_SIZE, LOCAL_RANK, NCCL_UID_PATH from environment.
// If WORLD_SIZE unset or == 1, returns single-GPU state (comm = nullptr).
// Calls cudaSetDevice(local_rank). Uses a shared file for ncclUniqueId exchange.
DistState dist_init_from_env();

void dist_destroy(DistState& st);

// In-place all-reduce SUM over a single buffer on the given stream.
// Safe to call with st.active() == false (no-op).
void dist_allreduce_sum(DistState& st, float* buf, size_t count, cudaStream_t stream);

// Group-mode helpers for batching many small all-reduces into a single NCCL op.
void dist_group_start(DistState& st);
void dist_group_end(DistState& st);

} // namespace rnet
