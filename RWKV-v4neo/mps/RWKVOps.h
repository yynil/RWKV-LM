#pragma once

// Defines the Metal soft shrink custom kernel.
static char *CUSTOM_KERNEL = R"WKV_OPS(
#include <metal_stdlib>
using namespace metal;

#define MIN_VALUE (-1e38)
#define Tmax 1024
kernel void wkv_forward(
                          constant int& B,
                          constant int& T,
                          constant int& C,
                          device const float* const _w,
                          device const float* const _u,
                          device const float* const _k,
                          device const float* const _v,
                          device float * const _y,
                          uint index [[thread_position_in_grid]]
                          ){
    const uint _b = index / C;
    const uint _c = index % C;
    const uint _offset = _b * T * C + _c;
    
    float u = _u[_c];
    float w = _w[_c];
    device const float* k = _k + _offset;
    device const float* v = _v + _offset;
    device float* y = _y + _offset;

    // aa and bb are running sums divided by exp(pp) (to avoid overflow)
    float aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = 0; i < T; i++) {
        const uint ii = i * C;
        const float kk = k[ii];
        const float vv = v[ii];

        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
        
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
}

kernel void rwkv_backward(constant int& B,
                            constant int& T,
                            constant int& C,
                            device const float* const _w,
                            device const float* const _u,
                            device const float* const _k,
                            device const float* const _v,
                            device const float * const _y,
                            device float * const _gy,
                            device float * const _gw,
                            device float * const _gu,
                            device float * const _gk,
                            device float * const _gv,
                            uint index [[ thread_position_in_grid ]]) {
    const uint _b = index / (C);
    const uint _c = index % (C);
    const uint _offset = _b * (T) * (C) + _c;
    float u = _u[_c];
    float w = _w[_c];
    device float const * const k = _k + _offset;
    device float const * const v = _v + _offset;
    device float const * const y = _y + _offset;
    device float const * const gy = _gy + _offset;
    device float * const gk = _gk + _offset;
    device float * const gv = _gv + _offset;
    float q[Tmax], r[Tmax];
    float gw = 0, gu = 0, aa = 0,bb=0,ga = 0,gb = 0,pp = MIN_VALUE;
    for(int i = 0;i < T;i ++){
        int const ii = i* (C);
        float const kk = k[ii];
        float const vv = v[ii];
        float const yy = y[ii];
        float ww = u + kk;
        float p = max(pp,ww);
        float e1 = exp(pp-p);
        float e2 = exp(ww-p);
        float const qq = gy[ii]/(e1*bb+e2);
        gw += (ga - gb * yy)*e1*qq;
        gu += (vv-yy)*e2*qq;
        q[i] = qq;
        r[i] = ww - p;
        ww = w + pp;
        p = max(ww,kk);
        e1 = exp(ww-p);
        e2 = exp(kk - p);
        ga = e1 * (aa + ga);
        gb = e1 * (bb + gb);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    int const _offsetBC = _b* (C) + _c;
    _gw[_offsetBC] = gw * _w[_c]; // multiply by w because of w -> -exp(w) in python forward()
    _gu[_offsetBC] = gu;

    aa = 0, bb = 0, pp = MIN_VALUE;
    for (int i = T - 1; i >= 0; i--) {
        const int ii = i * (C);
        const float kk = k[ii];
        const float vv = v[ii];
        const float yy = y[ii];
        const float qq = q[i];
        const float rr = r[i];

        float e1 = qq * exp(rr);
        float e2 = exp(kk + pp);
        gk[ii] = e1 * (vv - yy) + e2 * (aa * vv + bb);
        gv[ii] = e1 + e2 * aa;

        const float ww = w + pp;
        const float www = rr - u - kk;
        const float p = max(ww, www);
        e1 = exp(ww - p);
        e2 = qq * exp(www - p);
        aa = e1 * aa + e2;
        bb = e1 * bb - e2 * yy;
        pp = p;
    }
}

)WKV_OPS";
