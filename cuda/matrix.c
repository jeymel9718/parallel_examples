#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

#define WIDTH 16

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = 0;

	for (int k = 0; k<width; k++)
	{
		float Mdelement = Md[ty*width + k];
		float Ndelement = Nd[k*width + tx];
		Pvalue += Mdelement * Ndelement;
	}
	Pd[ty*width + tx] = Pvalue;
}

int main(void)
{
	float M[16][16], N[16][16], P[16][16];
	int Width = 16;
	int NUM = 192;
	// Inicializar datos de muestra
	for (int i = 0; i<16; i++)
	{
		for (int j = 0; j<16; j++)
		{
			M[i][j] = 2.0;
			N[i][j] = 3.0;
		}
	}
	int size = Width*Width*sizeof(float);

	float *Md, *Nd, *Pd;
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock(WIDTH, WIDTH);
	dim3 dimGrid(1, 1);
	MatrixMulKernel <<<dimGrid, dimBlock >> >(Md, Nd, Pd, Width);

	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

	// Imprime la matriz de resultados
	for (int i = 0; i<16; i++)
	{
		for (int j = 0; j<16; j++)
		{
			printf("%.2f  ", P[i][j]);
		}
		printf("\n");
	}

	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

	return 0;
}
