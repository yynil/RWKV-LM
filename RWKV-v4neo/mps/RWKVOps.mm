#include <torch/extension.h>
#include "RWKVOps.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

void check_device_and_contiguous(torch::Tensor t, char const * const info) {
    TORCH_CHECK(t.device().is_mps(), info , " must be a MPS tensor");
    TORCH_CHECK(t.is_contiguous(), info , " must be contiguous");
}

torch::Tensor& dispatchRwkvForward(int B,int T,int C,torch::Tensor& w,torch::Tensor& u,torch::Tensor& k,torch::Tensor& v,torch::Tensor& y) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = B*C;

        // Load the custom soft shrink shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        id<MTLFunction> rwkvFunction = [customKernelLibrary newFunctionWithName:@"wkv_forward"];
        TORCH_CHECK(rwkvFunction, "Failed to create function state object for wkv_forward");

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> rwkvPSO = [device newComputePipelineStateWithFunction:rwkvFunction error:&error];
        TORCH_CHECK(rwkvPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:rwkvPSO];
            [computeEncoder setBytes:&B length:sizeof(int) atIndex:0];
            [computeEncoder setBytes:&T length:sizeof(int) atIndex:1];
            [computeEncoder setBytes:&C length:sizeof(int) atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];
            [computeEncoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:7];
            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadExecutionWidth = rwkvPSO.threadExecutionWidth;
            if (threadExecutionWidth > B*C)
            {
                threadExecutionWidth = B*C;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadExecutionWidth, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return y;
}

void dispatchRwkvBackward(
        int B,int T,int C,
        torch::Tensor& w,
        torch::Tensor& u,
        torch::Tensor& k,
        torch::Tensor& v,
        torch::Tensor& y,
        torch::Tensor& gy,
        torch::Tensor& gw,
        torch::Tensor& gu,
        torch::Tensor& gk,
        torch::Tensor& gv
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        // Set the number of threads equal to the number of elements within the input tensor.
        int numThreads = B*C;

        // Load the custom soft shrink shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        id<MTLFunction> rwkvFunction = [customKernelLibrary newFunctionWithName:@"rwkv_backward"];
        TORCH_CHECK(rwkvFunction, "Failed to create function state object for rwkv_backward");

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> rwkvPSO = [device newComputePipelineStateWithFunction:rwkvFunction error:&error];
        TORCH_CHECK(rwkvPSO, error.localizedDescription.UTF8String);

        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Encode the pipeline state object and its parameters.
            [computeEncoder setComputePipelineState:rwkvPSO];
            [computeEncoder setBytes:&B length:sizeof(int) atIndex:0];
            [computeEncoder setBytes:&T length:sizeof(int) atIndex:1];
            [computeEncoder setBytes:&C length:sizeof(int) atIndex:2];
            [computeEncoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];
            [computeEncoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:7];
            [computeEncoder setBuffer:getMTLBufferStorage(gy) offset:gy.storage_offset() * gy.element_size() atIndex:8];
            [computeEncoder setBuffer:getMTLBufferStorage(gw) offset:gw.storage_offset() * gw.element_size() atIndex:9];
            [computeEncoder setBuffer:getMTLBufferStorage(gu) offset:gu.storage_offset() * gu.element_size() atIndex:10];
            [computeEncoder setBuffer:getMTLBufferStorage(gk) offset:gk.storage_offset() * gk.element_size() atIndex:11];
            [computeEncoder setBuffer:getMTLBufferStorage(gv) offset:gv.storage_offset() * gv.element_size() atIndex:12];
            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

            // Calculate a thread group size.
            NSUInteger threadExecutionWidth = rwkvPSO.threadExecutionWidth;
            if (threadExecutionWidth > B*C)
            {
                threadExecutionWidth = B*C;
            }
            MTLSize threadgroupSize = MTLSizeMake(threadExecutionWidth, 1, 1);

            // Encode the compute command.
            [computeEncoder dispatchThreads:gridSize
                      threadsPerThreadgroup:threadgroupSize];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
    return;
}

// C++ op dispatching the Metal wkv_forward
torch::Tensor wkv_forward(int B,int T,int C,torch::Tensor& w,torch::Tensor u,torch::Tensor& k,torch::Tensor& v,torch::Tensor& y) {
  
    check_device_and_contiguous(u, "u");
    check_device_and_contiguous(w, "w");
    check_device_and_contiguous(k, "k");
    check_device_and_contiguous(v, "v");
    check_device_and_contiguous(y, "y");



    return dispatchRwkvForward(B,
                                    T,
                                    C,
                                    w,
                                    u,
                                    k,
                                    v,
                                    y);
}


void wkv_backward(int B,int T,int C,
    torch::Tensor& w,
    torch::Tensor& u,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& y,
    torch::Tensor& dy,
    torch::Tensor& dw,
    torch::Tensor& du,
    torch::Tensor& dk,
    torch::Tensor& dv){
    check_device_and_contiguous(u, "u");
    check_device_and_contiguous(w, "w");
    check_device_and_contiguous(k, "k");
    check_device_and_contiguous(v, "v");
    check_device_and_contiguous(y, "y");
    check_device_and_contiguous(dy, "dy");
    check_device_and_contiguous(dw, "dw");
    check_device_and_contiguous(du, "du");
    check_device_and_contiguous(dk, "dk");
    check_device_and_contiguous(dv, "dv");
    dispatchRwkvBackward(
        B,T,C,
        w,u,k,v,y,dy,
        dw,du,dk,dv
    );
}
// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward);
    m.def("wkv_backward", &wkv_backward);
}
