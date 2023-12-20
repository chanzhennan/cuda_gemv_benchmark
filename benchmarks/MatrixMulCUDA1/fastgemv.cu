#include "MatrixMulCUDA1/fastgemv.cuh"

__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum) {
  if (threadNum >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (threadNum >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (threadNum >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (threadNum >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (threadNum >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

// thread_per_block = blockDim.x
// blockDim.y <= SHARED_MEM_MAX_ROWS
__global__ void fastgemv(half* mat, half* vec, half* res, unsigned int n,
                         unsigned int k, unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  unsigned int block_dim_x = blockDim.x;
  unsigned int bid = (unsigned int)(blockIdx.x);

  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  unsigned int idx = (unsigned int)(row) * (n / 8);

  unsigned int remainder = n % (block_dim_x * 8);
  if (row >= k) return;

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      half2* vec_h1 = (half2*)&vec_val.x;
      half2* vec_h2 = (half2*)&vec_val.y;
      half2* vec_h3 = (half2*)&vec_val.z;
      half2* vec_h4 = (half2*)&vec_val.w;
      half2* mat_h1 = (half2*)&mat_val.x;
      half2* mat_h2 = (half2*)&mat_val.y;
      half2* mat_h3 = (half2*)&mat_val.z;
      half2* mat_h4 = (half2*)&mat_val.w;
      sum += __half2float(vec_h1->x) * __half2float(mat_h1->x);
      sum += __half2float(vec_h1->y) * __half2float(mat_h1->y);
      sum += __half2float(vec_h2->x) * __half2float(mat_h2->x);
      sum += __half2float(vec_h2->y) * __half2float(mat_h2->y);
      sum += __half2float(vec_h3->x) * __half2float(mat_h3->x);
      sum += __half2float(vec_h3->y) * __half2float(mat_h3->y);
      sum += __half2float(vec_h4->x) * __half2float(mat_h4->x);
      sum += __half2float(vec_h4->y) * __half2float(mat_h4->y);
    }
  }

  unsigned int j = start_idx + (num_per_thread >> 3) * block_dim_x;
  if (0 != remainder) {
    if ((unsigned int)(j * 8) < n) {
      half* vec_val = (half*)&vec4[j];
      float4* mat_val = reinterpret_cast<float4*>(mat + row * n);
      half* mat_val2 = (half*)&mat_val[j];

      for (unsigned int i = 0; i < 8; i++) {
        if ((unsigned int)(j * 8 + i) < n) {
          sum += __half2float(vec_val[i]) * __half2float(mat_val2[i]);
        }
      }
    }
  }

  __syncthreads();

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE)
            ? warpLevelSums[threadIdx.y][laneId]
            : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

template <typename T>
void GEMV1(T* dVecTrans, T* dMatTrans, T* dResTrans, int m, int n, int k) {
  int mat_height_ = n;
  int vec_height_ = k;

  int block_dim_x = 128;
  int block_dim_y = 4;

  unsigned int num_per_thread = vec_height_ / block_dim_x;
  dim3 grid_dim(1, (mat_height_ + block_dim_y - 1) / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);

  fastgemv<<<grid_dim, block_dim>>>(dMatTrans, dVecTrans, dResTrans,
                                    vec_height_, mat_height_, num_per_thread);
}

template void GEMV1<half>(half* dVecTrans, half* dMatTrans, half* dResTrans,
                          int m, int n, int k);
