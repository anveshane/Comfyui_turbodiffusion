#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <torch/python.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "common/common.hpp"
#include "gemm/launch.hpp"

void int8_gemm(
  at::Tensor const& A, at::Tensor const& A_S, 
  at::Tensor const& B, at::Tensor const& B_S, 
  torch::Tensor& C
) {

  
  static constexpr int swizzle_dir = 1;
  static constexpr int swizzle_size_log = 5;

  int k = B.size(1);
  int m = A.size(0);
  int n = B.size(0);

  switch (C.scalar_type()) {
    case torch::kHalf:{
        int8_gemm_<cutlass::half_t> (
            (int8_t*)A.data_ptr(), A_S.data_ptr<float>(),
            (int8_t*)B.data_ptr(), B_S.data_ptr<float>(), 
            (cutlass::half_t*)C.data_ptr(), 
            m, n, k, swizzle_dir, swizzle_size_log, at::cuda::getCurrentCUDAStream().stream()
        );
        break;
    }

    case torch::kBFloat16:{
        int8_gemm_<cutlass::bfloat16_t> (
                (int8_t*)A.data_ptr(), A_S.data_ptr<float>(),
                (int8_t*)B.data_ptr(), B_S.data_ptr<float>(), 
                (cutlass::bfloat16_t*)C.data_ptr(), 
                m, n, k, swizzle_dir, swizzle_size_log, at::cuda::getCurrentCUDAStream().stream()
            );
        break;
    }

    default: {
      std::cerr << "Observing: " << C.scalar_type() << " for the output datatype which is invalid";
      throw std::runtime_error("Unsupported output data type for int8 gemm.");
    }
  }

}

void register_gemm(pybind11::module_ &m) {
    m.def("gemm_cuda", &int8_gemm);
}


  


