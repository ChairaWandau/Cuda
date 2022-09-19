#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void Palindrom(int* a, int* arr, int count) {
	int m = 0;
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	while (*a > 0)
	{
		m = m * 10 + *a % 10;
		*a = *a / 10;
	}
	*a = m;
	for (int i = idx-1; i >= 0; i--) {
		arr[i] = *a % 10;
		*a /= 10;
	}
}
int main(void) {
	//Задаем значение числа, из которого будем получать полином (двузначное число)
	int A=96;
	int a=0; // host копия a
	int* dev_a; // device копия a
	//выделяем память для device копии a
	cudaMalloc((void**)&dev_a, sizeof(int));
	//До тех пор, пока число - не палиндром
	while (A != a) {
		A += a;
		a = A;
		// копируем ввод на device
		cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
		//считаем, сколько цифр в числе
		int count = 0;
		int n = A;
		while (n != 0)
		{
			n = n / 10;
			count++;
		}
		//создаем массив для цифр
		int* arr = new int[count];
		int* dev_arr;
		cudaMalloc((void**)&dev_arr, count*sizeof(int));
		// запускаем Palindrom() на GPU, передавая параметры
		Palindrom <<< count, 2 >>> (dev_a, dev_arr, count);
		cudaMemcpy(arr, dev_arr, sizeof(int) * count, cudaMemcpyDeviceToHost);
		//Превращаем массив в число
		a=0;
		for (int i = 0; i < count; i++)
			a = a * 10 + arr[i];
	}
	// освобождаем память
	cudaFree(dev_a);
	// выводим результат на экран
	printf("%i\n", A);
	return 0;
}
