#include <math.h>
// #include <torch/extension.h>
#include <torch/torch.h>
#include <iostream>
#include <c10/cuda/CUDAException.h>

void gelu_kernel(float* in, float* out, int num_elements){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < num_elements){
        out[i] = in[i] * 10;
    }
    printf("i %d in=%.3f out=%.3f\n", i, in[i], out[i]);

}

int main(){
    torch::Tensor in = torch::eye(3);
    torch::Tensor out = torch::empty_like(in);
    gelu_kernel(in.data_ptr<float>, out.data_ptr<float>, in.numel());
    printf("done");

    return 0;
}