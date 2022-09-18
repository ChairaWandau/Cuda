#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h> 

__global__ void Palindrom(int* a) {
	// перевернутое число A
	int m = 0;
	// Переворачиваем число
	while (*a > 0)
	{
		m = m * 10 + *a % 10;
		*a = *a / 10;
	}
	*a = m;
}
int main(void) {
	//Задаем значение числа, из которого будем получать полином (двузначное число)
	int A=10876;
	int a=0; // host копия a
	int* dev_a; // device копия a
	//До тех пор, пока число - не палиндром
	while (A != a) {
		A += a;
		a = A;
		//выделяем память для device копии a
		cudaMalloc((void**)&dev_a, sizeof(int));
		// копируем ввод на device
		cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
		// запускаем Palindrom() на GPU, передавая параметры
		Palindrom << < 1, 1 >> > (dev_a);
		// копируем результат функции обратно в a
		cudaMemcpy(&a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
	}
	// освобождаем память
	cudaFree(dev_a);
	// выводим результат на экран
	printf("%i\n", A);
	return 0;
}