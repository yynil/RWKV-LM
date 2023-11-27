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
        threadgroup float r[_N_], k[_N_], u[_N_], w[_N_];
        float state[_N_]={0};
                                  
                                  
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
}


kernel void rwkv5_backward(
                        constant uint& B [[buffer(0)]],
                        constant uint& T [[buffer(1)]],
                        constant uint& C [[buffer(2)]], 
                        constant uint& H [[buffer(3)]],
                        constant F * const _r [[buffer(4)]],
                        constant F * const _k [[buffer(5)]],
                        constant F * const _v [[buffer(6)]],
                        constant F *_w [[buffer(7)]],
                        constant F *__w [[buffer(8)]],
                        constant F *_u [[buffer(9)]],
                        constant F * const _gy [[buffer(10)]],
                        device F * const _gr [[buffer(11)]],
                        device F * const _gk [[buffer(12)]],
                        device F * const _gv [[buffer(13)]],
                        device F * const _gw [[buffer(14)]],
                        device F * const _gu [[buffer(15)]],
                        uint gid   [[ threadgroup_position_in_grid ]],
                        uint tid [[ thread_position_in_threadgroup ]]
                           )
{
    const uint b = gid / H;
    const uint h = gid % H;
    const uint i = tid;
    _w += h*_N_;
    _u += h*_N_;
    __w += h*_N_;

    threadgroup float w_[_N_], u_[_N_];
    threadgroup float r[_N_], k[_N_], v[_N_], gy[_N_];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    w_[i] = _w[i];
    u_[i] = float(_u[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float w = w_[i];
    const float ww = __w[i];
    const float u = u_[i];

    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    float gw = 0, gu = 0;
    const int t000 = b * T * C + h * _N_ + i;
    const int t111 = (b + 1) * T * C + h * _N_ + i;
    const int t222 = t111 - 2 * C;
    for (int t = t000; t < t111; t += C)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[i] = _v[t];
        gy[i] = _gy[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float k = _k[t];
        float gr = 0, gu_ = 0;

        for (int j = 0; j < _N_; j++)
        {
            thread float &s = state[j];
            float x = k * v[j];

            gr += (u * x + s) * gy[j];
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = gr;
        gu += _r[t] * gu_;
    }
    
    _gu[b * C + h * _N_ + i] = F(gu);

    for (int t = t000; t < t222; t += C)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[i] = _v[t];
        gy[i] = _gy[t + 2 * C];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float k = _k[t];
        float gw_ = 0;

        for (int j = 0; j < _N_; j++)
        {
            thread float &s = saaaa[j];
            thread float &s2 = sbbbb[j];
            float x = k * v[j];

            float tmp = w * (x + s);
            s = tmp;
            s2 = tmp + w * s2;
            gw_ += s2 * gy[j];
        }
        gw += _r[t + 2 * C] * gw_;
    }
    _gw[b * C + h * _N_ + i] = F(ww * gw);

    for (int t = t111 - C; t >= t000; t -= C)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[i] = _v[t];
        gy[i] = _gy[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float rr = _r[t];
        float gk = 0;

        for (int j = 0; j < _N_; j++)
        {
            thread float &s = scccc[j];
            float x = rr * gy[j];

            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = gk;
    }
    
    for (int t = t111 - C; t >= t000; t -= C)
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        r[i] = _r[t];
        k[i] = _k[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float gyy = _gy[t];
        float gv = 0;

        for (int j = 0; j < _N_; j++)
        {
            thread float &s = sdddd[j];
            float x = gyy * r[j];

            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = gv;
    }
    
}
)WKV_OPS";
