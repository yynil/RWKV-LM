#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(
    const uint B,
    const uint T,
    const uint C,
    const uint H,
    float* _state,
    const F* _r,
    const F* _k,
    const F* _v,
    const float* _w,
    const F* _u,
    F* _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    _w += h*_N_;
    _u += h*_N_;
    _state += b*H*_N_*_N_ + h*_N_*_N_  + i*_N_; 

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_];
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j];  

    __syncthreads();
    w[i] = _w[i];
    u[i] = _u[i];
    __syncthreads();

    for (uint t =  b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        r[i] = _r[t];
        k[i] = _k[t];
        __syncthreads();

        const float v = _v[t];
        float y = 0;
        #pragma unroll
        for (uint j = 0; j < _N_; j+=4)
        {
            float4 r_ = make_float4(r[j], r[j+1], r[j+2], r[j+3]);
            float4 k_ = make_float4(k[j], k[j+1], k[j+2], k[j+3]);
            float4 w_ = make_float4(w[j], w[j+1], w[j+2], w[j+3]);
            float4 u_ = make_float4(u[j], u[j+1], u[j+2], u[j+3]);
            float4 s = make_float4(state[j], state[j+1], state[j+2], state[j+3]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
            state[j] = s.x;
            state[j+1] = s.y;
            state[j+2] = s.z;
            state[j+3] = s.w;
        }
        _y[t] = F(y);
    }
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];
}

void cuda_forward(int B, int T, int C, int H, float* state,bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, state, r, k, v, w, u, y);
}
