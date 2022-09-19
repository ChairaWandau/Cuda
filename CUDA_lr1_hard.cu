#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
__global__ void Dispersion(float *a, int* m, float* dispersion) {
	// Мат ожидание
	float MX = 0;
	// Мат ожидание от X^2
	float MX_2 = 0;
	// Перебираем массив, находя мат ожидания
	for (int j = 0; j < *m; j++)
	{
		MX += a[0 * *m + j] * a[1 * *m + j];
		MX_2 += a[0 * *m + j]* a[0 * *m + j] * a[1 * *m + j];
	}
	//Считаем дисперсию
	*dispersion = MX_2 - MX * MX;
}
int main(void) {
	int i, j; // указатели на элемент массива
	int n=2, m; // количество строк и столбцов
	int* dev_m; // device копия m
	float dispersion;  // результат расчета дисперсии
	float* dev_dispersion;  // device копия результата расчета дисперсии
	printf("Enter number of columns: ");
	scanf("%d", &m);
	float* a = new float[n * m];  // указатель на массив
	float* dev_a = new float[n * m]; // device копия a
	// Ввод элементов массива
	for (i = 0; i < n; i++)  // цикл по строкам
	{
		for (j = 0; j < m; j++)  // цикл по столбцам
		{
			printf("a[%d][%d] = ", i, j);
			scanf("%f", (float*)(a + i * m + j));
		}
	}
	//выделяем память
	cudaMalloc((void**)&dev_a, n*m*sizeof(float));
	cudaMalloc((void**)&dev_m, sizeof(int));
	cudaMalloc((void**)&dev_dispersion, sizeof(float));
	// копируем ввод на device
	cudaMemcpy(dev_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_m, &m, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dispersion, &dispersion, sizeof(float), cudaMemcpyHostToDevice);
	// запускаем Dispersion() на GPU, передавая параметры
	Dispersion << < 1, 1 >> > (dev_a, dev_m, dev_dispersion);
	// копируем результат функции обратно
	cudaMemcpy(&dispersion, dev_dispersion, sizeof(float), cudaMemcpyDeviceToHost);
	// освобождаем память
	cudaFree(dev_a);
	cudaFree(dev_m);
	cudaFree(dev_dispersion);
	// выводим результат на экран
	printf("D = %f\n", dispersion);
	return 0;
}