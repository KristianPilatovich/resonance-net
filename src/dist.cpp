#include "dist.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <sys/stat.h>

#ifdef RNET_NCCL
#include <nccl.h>
#endif

namespace rnet {

#ifdef RNET_NCCL
#define NCCL_CHECK(call) do { \
    ncclResult_t r = (call); \
    if (r != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        std::abort(); \
    } \
} while (0)
#endif

static int env_int(const char* name, int defval) {
    const char* s = std::getenv(name);
    if (!s || !*s) return defval;
    return std::atoi(s);
}

DistState dist_init_from_env() {
    DistState st;
    st.world_size = env_int("WORLD_SIZE", 1);
    st.rank       = env_int("RANK", 0);
    st.local_rank = env_int("LOCAL_RANK", st.rank);

    if (st.world_size <= 1) {
        cudaSetDevice(st.local_rank);
        return st;
    }

#ifndef RNET_NCCL
    fprintf(stderr, "WORLD_SIZE=%d but binary built without RNET_NCCL\n", st.world_size);
    std::abort();
#else
    cudaError_t ce = cudaSetDevice(st.local_rank);
    if (ce != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
                st.local_rank, cudaGetErrorString(ce));
        std::abort();
    }

    const char* uid_path = std::getenv("NCCL_UID_PATH");
    if (!uid_path) {
        fprintf(stderr, "NCCL_UID_PATH env var not set\n");
        std::abort();
    }

    ncclUniqueId uid;
    if (st.rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&uid));
        std::string tmp = std::string(uid_path) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "Cannot write uid to %s\n", tmp.c_str());
            std::abort();
        }
        std::fwrite(&uid, sizeof(uid), 1, f);
        std::fclose(f);
        // Atomic rename so waiters only see a complete file.
        std::rename(tmp.c_str(), uid_path);
    } else {
        using clock = std::chrono::steady_clock;
        auto deadline = clock::now() + std::chrono::seconds(120);
        struct stat sb;
        while (stat(uid_path, &sb) != 0 || (size_t)sb.st_size < sizeof(uid)) {
            if (clock::now() > deadline) {
                fprintf(stderr, "rank %d: timeout waiting for %s\n", st.rank, uid_path);
                std::abort();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        FILE* f = std::fopen(uid_path, "rb");
        if (!f) {
            fprintf(stderr, "Cannot read uid from %s\n", uid_path);
            std::abort();
        }
        if (std::fread(&uid, sizeof(uid), 1, f) != 1) {
            fprintf(stderr, "Short read on uid file %s\n", uid_path);
            std::abort();
        }
        std::fclose(f);
    }

    NCCL_CHECK(ncclCommInitRank(&st.comm, st.world_size, uid, st.rank));
    return st;
#endif
}

void dist_destroy(DistState& st) {
#ifdef RNET_NCCL
    if (st.comm) {
        ncclCommDestroy(st.comm);
        st.comm = nullptr;
    }
#endif
    st.world_size = 1;
}

void dist_allreduce_sum(DistState& st, float* buf, size_t count, cudaStream_t stream) {
    if (!st.active()) return;
#ifdef RNET_NCCL
    NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, st.comm, stream));
#endif
}

void dist_group_start(DistState& st) {
#ifdef RNET_NCCL
    if (st.active()) NCCL_CHECK(ncclGroupStart());
#endif
}

void dist_group_end(DistState& st) {
#ifdef RNET_NCCL
    if (st.active()) NCCL_CHECK(ncclGroupEnd());
#endif
}

} // namespace rnet
