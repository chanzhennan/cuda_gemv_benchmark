file(GLOB_RECURSE SRCS *.cu *.h *.hpp *.cc)

if(NOT BUILD_SHARED_LIBS)
  cuda_add_library(bm_base ${SRCS})
else()
  cuda_add_library(bm_base SHARED ${SRCS})
endif()
target_link_libraries(bm_base benchmark::benchmark pthread bmlib)
cuda_add_cublas_to_target(bm_base)
