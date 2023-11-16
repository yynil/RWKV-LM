#include <torch/extension.h>
#include "RWKV5Ops.h"

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

torch::Tensor& dispatchRwkv5Forward(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor& r,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& w,
    torch::Tensor& u,
    torch::Tensor& y) {
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

        id<MTLFunction> rwkvFunction = [customKernelLibrary newFunctionWithName:@"rwkv5_forward"];
        TORCH_CHECK(rwkvFunction, "Failed to create function state object for rwkv5_forward");

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
            [computeEncoder setBytes:&H length:sizeof(int) atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(r) offset:r.storage_offset() * r.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];
            [computeEncoder setBuffer:getMTLBufferStorage(w) offset:w.storage_offset() * w.element_size() atIndex:7];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:8];
            [computeEncoder setBuffer:getMTLBufferStorage(y) offset:y.storage_offset() * y.element_size() atIndex:9];
            MTLSize threads = MTLSizeMake(C/H, 1, 1);
            MTLSize groups = MTLSizeMake(B * H, 1, 1);
            NSLog(@"threads: %d, %d, %d", threads.width, threads.height, threads.depth);
            NSLog(@"groups: %d, %d, %d", groups.width, groups.height, groups.depth);
            
            // Encode the compute command.
            [computeEncoder dispatchThreadgroups:groups
                      threadsPerThreadgroup:threads];

            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }

    return y;
}


// C++ op dispatching the Metal wkv_forward
torch::Tensor wkv5_forward(
            int B,int T,int C,int H,
            torch::Tensor& r,
            torch::Tensor& k,
            torch::Tensor& v,
            torch::Tensor& w,
            torch::Tensor& u,
            torch::Tensor& y) {
  
    check_device_and_contiguous(u, "u");
    check_device_and_contiguous(w, "w");
    check_device_and_contiguous(r, "r");
    check_device_and_contiguous(k, "k");
    check_device_and_contiguous(v, "v");
    check_device_and_contiguous(y, "y");



    return dispatchRwkv5Forward(
                B,
                T,
                C,
                H,
                r,
                k,
                v,
                w,
                u,
                y);
}


void dispatchRwkv5Backward(
        int B,
        int T,
        int C,
        int H,
        torch::Tensor& r,
        torch::Tensor& k,
        torch::Tensor& v,
        torch::Tensor& eew,
        torch::Tensor& ew,
        torch::Tensor& u,
        torch::Tensor& dy,
        torch::Tensor& dr,
        torch::Tensor& dk,
        torch::Tensor& dv,
        torch::Tensor& dw,
        torch::Tensor& du
) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;


        // Load the custom soft shrink shader.
        id<MTLLibrary> customKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:CUSTOM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        id<MTLFunction> rwkvFunction = [customKernelLibrary newFunctionWithName:@"rwkv5_backward"];
        TORCH_CHECK(rwkvFunction, "Failed to create function state object for rwkv5_backward");

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
            [computeEncoder setBytes:&C length:sizeof(int) atIndex:3];
            [computeEncoder setBuffer:getMTLBufferStorage(r) offset:r.storage_offset() * r.element_size() atIndex:4];
            [computeEncoder setBuffer:getMTLBufferStorage(k) offset:k.storage_offset() * k.element_size() atIndex:5];
            [computeEncoder setBuffer:getMTLBufferStorage(v) offset:v.storage_offset() * v.element_size() atIndex:6];
            [computeEncoder setBuffer:getMTLBufferStorage(eew) offset:eew.storage_offset() * eew.element_size() atIndex:7];
            [computeEncoder setBuffer:getMTLBufferStorage(ew) offset:ew.storage_offset() * ew.element_size() atIndex:8];
            [computeEncoder setBuffer:getMTLBufferStorage(u) offset:u.storage_offset() * u.element_size() atIndex:9];
            [computeEncoder setBuffer:getMTLBufferStorage(dy) offset:dy.storage_offset() * dy.element_size() atIndex:10];
            [computeEncoder setBuffer:getMTLBufferStorage(dr) offset:dr.storage_offset() * dr.element_size() atIndex:11];
            [computeEncoder setBuffer:getMTLBufferStorage(dk) offset:dk.storage_offset() * dk.element_size() atIndex:12];
            [computeEncoder setBuffer:getMTLBufferStorage(dv) offset:dv.storage_offset() * dv.element_size() atIndex:13];
            [computeEncoder setBuffer:getMTLBufferStorage(dw) offset:dw.storage_offset() * dw.element_size() atIndex:14];
            [computeEncoder setBuffer:getMTLBufferStorage(du) offset:du.storage_offset() * du.element_size() atIndex:15];
            MTLSize threads = MTLSizeMake(C/H, 1, 1);
            MTLSize groups = MTLSizeMake(B * H, 1, 1);
            NSLog(@"threads: %d, %d, %d", threads.width, threads.height, threads.depth);
            NSLog(@"groups: %d, %d, %d", groups.width, groups.height, groups.depth);

            // Encode the compute command.
            [computeEncoder dispatchThreadgroups:groups
                      threadsPerThreadgroup:threads];
            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
    return;
}

void wkv5_backward(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor& r,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& ew,
    torch::Tensor& w,
    torch::Tensor& u,
    torch::Tensor& dy,
    torch::Tensor& dr,
    torch::Tensor& dk,
    torch::Tensor& dv,
    torch::Tensor& dw,
    torch::Tensor& du){
    check_device_and_contiguous(r, "r");
    check_device_and_contiguous(k, "k");
    check_device_and_contiguous(v, "v");
    check_device_and_contiguous(u, "u");
    check_device_and_contiguous(w, "w");
    check_device_and_contiguous(ew, "ew");
    check_device_and_contiguous(dy, "dy");
    check_device_and_contiguous(dk, "dk");
    check_device_and_contiguous(dv, "dv");
    check_device_and_contiguous(dw, "dw");
    check_device_and_contiguous(du, "du");
    dispatchRwkv5Backward(
        B,T,C,H,
        r,k,v,w,ew,u,
        dy,dr,dk,dv,dw,du
    );
}
// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv5_forward", &wkv5_forward);
    m.def("wkv5_backward", &wkv5_backward);
}
