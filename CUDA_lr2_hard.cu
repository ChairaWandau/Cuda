#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define N 10

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generate(curandState* globalState, float* randomArray)
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform(&localState);
    randomArray[ind] = RANDOM;
    globalState[ind] = localState;
}

int main(int argc, char** argv)
{
    dim3 T(N, 1, 1);
    curandState* devStates;
    //массив рандомных чисел
    float* randomValues = new float[N];
    //копия массива рандомных чисел
    float* devRandomValues;
    //выделяем память
    cudaMalloc(&devStates, N * sizeof(curandState));
    cudaMalloc(&devRandomValues, N * sizeof(*randomValues));
    //устанавливаем seeds
    setup_kernel << <1, T >> > (devStates, time(NULL));
    //Генерируем рандомные числа
    generate << <1, T >> > (devStates, devRandomValues);
    //копируем значения
    cudaMemcpy(randomValues, devRandomValues, N * sizeof(*randomValues), cudaMemcpyDeviceToHost);
    //выводим на экран
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", randomValues[i]);
    }
    //освобождаем память
    cudaFree(devRandomValues);
    cudaFree(devStates);
    delete randomValues;
    getchar();
    return 0;
}