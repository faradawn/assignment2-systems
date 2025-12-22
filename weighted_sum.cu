#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void vecMult(float* w, float* x, float* res)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    res[tid] = w[tid] * x[tid];
}

__global__ void vecSum(float* vec)
{
    int tid = threadIdx.x;
    __shared__ float finalVec[8192];
    
    // Load data into shared memory
    finalVec[tid] = vec[tid];
    finalVec[tid + blockDim.x] = vec[tid + blockDim.x];
    __syncthreads();
    
    for(int stride = 8192/2; stride > 0; stride >>= 1){
        if(tid >= stride) break;

        finalVec[tid] = finalVec[tid] + finalVec[tid + stride];
        __syncthreads();
    }
    
    vec[tid] = finalVec[tid];
}

int main()
{
    int vectorLength = 8192;
    int vectorSize = vectorLength * sizeof(float);

    float *x, *w, *res;
    cudaMallocHost(&x, vectorSize);
    cudaMallocHost(&w, vectorSize);
    cudaMallocHost(&res, vectorSize);

    // init the values
    std::fill(x, x + vectorLength, 4.0f); 
    std::fill(w, w + vectorLength, 0.5f);  

    float *d_x, *d_w, *d_res;
    cudaMalloc(&d_x, vectorSize);
    cudaMalloc(&d_w, vectorSize);
    cudaMalloc(&d_res, vectorSize);

    // copy to GPU
    cudaMemcpy(d_w, w, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, vectorSize, cudaMemcpyHostToDevice);

    // why cannot 
    // for(int i = 0; i < vectorLength; i++){
    //     d_x[i] = x[i];
    // }


    // step 1: element-wise vector product
    vecMult<<<1, 8192>>>(d_w, d_x, d_res);
    cudaDeviceSynchronize();

    // step 2: vector sum (reduce)
    vecSum<<<1, 8192/2>>>(d_res);

    cudaMemcpy(res, d_res, vectorSize, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 5; i++){
        printf("%f ", res[i]);
    }
    
    // ✅ Cleanup
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_res);
    cudaFreeHost(x);
    cudaFreeHost(w);
    cudaFreeHost(res);
}