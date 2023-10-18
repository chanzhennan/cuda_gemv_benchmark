add_subdirectory(bm_lib)

set(SOURCE_FILES cuda_benchmark.cc)

list(
  APPEND
  SOURCE_FILES
  MatrixMulCUDA0/naive.cc
  MatrixMulCUDA0/naive.cu
  )

cuda_add_executable(cuda_benchmark ${SOURCE_FILES})
set_property(TARGET cuda_benchmark PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(cuda_benchmark bm_lib cublas)

install(TARGETS cuda_benchmark RUNTIME)
