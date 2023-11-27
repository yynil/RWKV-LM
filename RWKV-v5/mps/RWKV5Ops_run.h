#pragma once

// Defines the Metal soft shrink custom kernel.
static char *CUSTOM_KERNEL = R"WKV_OPS(
#include <metal_stdlib>
#define _N_ 64
using namespace metal;
#define F float
kernel void rwkv5_forward(
                          constant uint& B,
                          constant uint& T,
                          constant uint& C,
                          constant uint& H,
                          device F* _state,
                          device const F* const _r,
                          device const F* const _k,
                          device const F* const _v,
                          device const float* _w,
                          device const F* _u,
                          device F * const _y,
                          uint gid   [[ threadgroup_position_in_grid ]],
                          uint tid [[ thread_position_in_threadgroup ]]){

        const uint b = gid/ H;
        const uint h = gid % H;
        const uint i = tid;
        _w += h*_N_;
        _u += h*_N_;
        _state += h*_N_*_N_ + i*_N_; // wrong if B > 1 !!!
        threadgroup float r[_N_], k[_N_], u[_N_], w[_N_];
        float state[_N_];
        for (int j = 0; j < _N_; j++)
            state[j] = _state[j];  
                                  
                                  
        threadgroup_barrier(mem_flags::mem_threadgroup);
        w[i] = _w[i];
        u[i] = float(_u[i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
                              
      for (uint t =  b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
          {
              threadgroup_barrier(mem_flags::mem_threadgroup);
              r[i] = _r[t];
              k[i] = _k[t];
              threadgroup_barrier(mem_flags::mem_threadgroup);

              const float v = float(_v[t]);
              float y = 0;

              for (uint j = 0; j < _N_; j+=4)
              {
                  const float4 r_ = float4(r[j], r[j+1], r[j+2], r[j+3]);
                  const float4 k_ = float4(k[j], k[j+1], k[j+2], k[j+3]);
                  const float4 w_ = float4(w[j], w[j+1], w[j+2], w[j+3]);
                  const float4 u_ = float4(u[j], u[j+1], u[j+2], u[j+3]);
                  float4 s = float4(state[j], state[j+1], state[j+2], state[j+3]);
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

)WKV_OPS";
