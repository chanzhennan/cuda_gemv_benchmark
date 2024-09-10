add_subdirectory(bm_base)

set(SOURCE_FILES cuda_benchmark.cc)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -std=c++17)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-line-info
                    -arch=sm_${CUDA_ARCH})
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas=-v -U__CUDA_NO_HALF_OPERATORS__)

list(
  APPEND
  SOURCE_FILES
  MatrixMulCUDA0/naive.cc
  MatrixMulCUDA0/naive.cu
  MatrixMulCUDA4/cublas.cc
  MatrixMulCUDA1/fastgemv.cc
  MatrixMulCUDA1/fastgemv.cu
  # MatrixMulCUDA3/fastgemv_inst.cc MatrixMulCUDA3/fastgemv_inst.cu
)

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
cuda_add_cublas_to_target(cuda_benchmark)

set_property(TARGET cuda_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(cuda_benchmark bm_base bmlib)

install(TARGETS cuda_benchmark RUNTIME)
