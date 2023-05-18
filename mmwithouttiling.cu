#include <stdio.h>
#include <cuda_runtime.h>
//kernel function
__global__ void matrix_multiplication(float *a, float *b, float *c, int M, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += a[i * N + k] * b[k * N + j];
        }
        c[i * N + j] = sum;
    }
}
int main(){
    int M = 1000;
    int N = 500;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size_a = M * N * sizeof(float);
    int size_b = N * N * sizeof(float);
    int size_c = M * N * sizeof(float);
    a = (float*) malloc(size_a);
    b = (float*) malloc(size_b);
    c = (float*) malloc(size_c);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i * N + j] = i - j;
        }
    }
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
    dim3 dimGrid((M + 15) / 16, (N + 15) / 16, 1);
    dim3 dimBlock(16, 16, 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }
    printf("Execution time: %.2f ms\n", milliseconds);
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}