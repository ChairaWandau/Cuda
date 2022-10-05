#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 32

__global__ void MatrixMultiplication(double *A, double *B, double *C, int *Arow, int* Acolumn, int *Bcolumn)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (r < *Arow && c < *Bcolumn)
	{
		C[r * (*Bcolumn) + c] = 0;
		for (int k = 0; k < *Acolumn; k++)
		{
			C[r * (*Bcolumn) + c] += A[r * (*Acolumn) + k] * B[k * (*Bcolumn)  + c];
		}
	}
}
int main(void) {
	//Матрицы
	double *A, *B, *C;
	//Копии матриц
	double *dev_A, *dev_B, *dev_C;
	//Размеры матриц
	int Arow=0;
	int Acolumn=0;
	int Bcolumn=0;
	//Копии размеров матриц
	int *dev_Arow, * dev_Acolumn,*dev_Bcolumn;
	printf("Enter the number of rows of matrix A: ");
	scanf("%d", &Arow);
	printf("Enter the number of columns of matrix A (number of rows of matrix B): ");
	scanf("%d", &Acolumn);
	printf("Enter the number of columns of matrix B: ");
	scanf("%d", &Bcolumn);
	//Выделение памяти для матриц
	A = (double*)malloc(Arow * Acolumn * sizeof(double));
	B = (double*)malloc(Acolumn * Bcolumn * sizeof(double));
	C = (double*)malloc(Arow * Bcolumn * sizeof(double));
	//Заполнение матриц рандомными стозначными числами
	printf("============================================\nMatrix A:\n");
	for (int i = 0; i < Arow; i++)
	{
		for (int j = 0; j < Acolumn; j++)
		{
			//числа от 1 до 10, умноженные на 10^100
			A[i+j* Arow] = (1 + rand() % 10) * pow(10, 100);
			printf("A[%d][%d]=%lf\n", i, j, A[i + j * Arow]);
		}
	}
	printf("============================================\nMatrix B:\n");
	for (int i = 0; i < Acolumn; i++)
	{
		for (int j = 0; j < Bcolumn; j++)
		{
			//числа от 1 до 10, умноженные на 10^100
			B[i + j * Acolumn] = (1 + rand() % 10) * pow(10, 100);
			printf("B[%d][%d]=%lf\n", i, j, B[i+j * Acolumn]);
		}
	}
	//Все нужное для подсчета времени работы
	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//начало отсчета времени
	cudaEventRecord(start, 0);
	//Выделение памяти для копий
	cudaMalloc((void**)&dev_A, Arow * Acolumn * sizeof(double));
	cudaMalloc((void**)&dev_B, Acolumn * Bcolumn * sizeof(double));
	cudaMalloc((void**)&dev_C, Arow * Bcolumn * sizeof(double));
	cudaMalloc((void**)&dev_Arow, sizeof(int));
	cudaMalloc((void**)&dev_Acolumn, sizeof(int));
	cudaMalloc((void**)&dev_Bcolumn, sizeof(int));
	//копирование на GPU
	cudaMemcpy(dev_A, A, Arow * Acolumn * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, Acolumn * Bcolumn * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Arow, &Arow, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Acolumn, &Acolumn, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Bcolumn, &Bcolumn, sizeof(int), cudaMemcpyHostToDevice);
	//Определение количества потоков и блоков потоков
	dim3 dim_grid(ceilf(Arow / (float)BLOCK_SIZE), ceilf(Bcolumn / (float)BLOCK_SIZE), 1);
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
	//Запуск умножения матриц
	MatrixMultiplication <<<dim_grid, dim_block >>> (dev_A, dev_B, dev_C, dev_Arow, dev_Acolumn, dev_Bcolumn);
	//Копирование результата расчета
	cudaMemcpy(C, dev_C, Arow * Bcolumn * sizeof(double), cudaMemcpyDeviceToHost);
	//Конец отсчета времени
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	//вывод результата перемножения на экран
	printf("============================================\nGPU Matrix C:\n");
	for (int i = 0; i < Arow; i++)
	{
		for (int j = 0; j < Bcolumn; j++)
		{
			printf("C[%d][%d]=%lf\n", i, j, C[i+j * Arow]);
		}
	}
	//Освобождение памяти
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaFree(dev_Arow);
	cudaFree(dev_Bcolumn);
	//вывод времени работы программы на GPU
	printf("============================================\nGPU time: %.2f milliseconds\n", gpuTime);
	//Блок расчета на CPU
	int start1, time1;
	start1 = clock();
	printf("============================================\nCPU Matrix C:\n");
	for (int i = 0; i < Arow; i++)
	{
		for (int j = 0; j < Bcolumn; j++)
		{
			C[i * Bcolumn + j] = 0.0;
			for (int k = 0; k < Acolumn; k++)
			{
				C[i * Bcolumn + j] += A[i * Acolumn + k] * B[k * Bcolumn + j];
			}
			printf("C[%d][%d]=%lf\n", i, j, C[i + j * Arow]);
		}
	}
	time1 = clock() - start1;
	float cpuTime = time1 / 2.0;
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	//вывод времени работы программы на CPU
	printf("============================================\nCPU time: %.2f milliseconds\n", cpuTime);
	return 0;
}