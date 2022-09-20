#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N (256)
__global__ void Palindrom(int* a) {
	int idx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
	int A = threadIdx.x*10;
	int B = 0;
	while (A != B)
	{
		A += B;
		B = A;
		int m = 0;
		while (B > 0)
		{
			m = m * 10 + B % 10;
			B = B / 10;
		}
		B = m;
	}
	a[idx] = A;
}
int main(void) {
	int a[N];
	int* dev = NULL;
	cudaMalloc((void**)&dev, N * sizeof(int));
	// запускаем Palindrom() на GPU, передавая параметры
	Palindrom << <dim3(N/16, 1), dim3(16,1) >> > (dev);
	// копируем результат функции обратно в a
	cudaMemcpy(a, dev, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dev);
	for (int idx = 0; idx < N; idx++) 
	{
		printf("a[%d]=%d\n", idx, a[idx]);
	}
	return 0;
}
