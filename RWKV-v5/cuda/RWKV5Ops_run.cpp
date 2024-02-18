#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
void cuda_forward(int B, int T, int C, int H, float* state,bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y);


void forward(
            int64_t B, int64_t T, int64_t C, int64_t H, 
            torch::Tensor &state, 
            torch::Tensor &r, 
            torch::Tensor &k, 
            torch::Tensor &v, 
            torch::Tensor &w, 
            torch::Tensor &u, 
            torch::Tensor &y) {
    cuda_forward(B, T, C, H, state.data_ptr<float>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<bf16>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv5_forward", &forward, "wkv5 forward");
}

TORCH_LIBRARY(wkv5, m) {
    m.def("wkv5_forward", forward);
}
