#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_quant(py::module_ &);
void register_rms_norm(pybind11::module_ &);
void register_layer_norm(pybind11::module_ &);
void register_gemm(pybind11::module_ &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_quant(m);
    register_rms_norm(m);
    register_layer_norm(m);
    register_gemm(m);
}