#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "error.cuh"

//Dimensions of the first matrix
#define ROWS1 10000
#define COLS1 10000
#define SIZE1 ROWS1*COLS1

//Dimensions of the seconds matrix
#define ROWS2 10000
#define COLS2 10000
#define SIZE2 ROWS2*COLS2

#define SIZE3 ROWS1*COLS2
#define BLOCK_SIZE 16


/* Function to do matrix multiplication */
__global__ void matMul(int * matC_cuda, int * matA_cuda, int * matB_cuda){

	int row_index=blockDim.y*blockIdx.y+threadIdx.y;
    int col_index=blockDim.x*blockIdx.x+threadIdx.x; 

	if(row_index<ROWS1 && col_index<COLS2){
		int prod=0;
		int k;
		for(k=0;k<COLS1;k++){
			prod=prod+matA_cuda[row_index*COLS1+k]*matB_cuda[k*COLS2+col_index];
		}
		matC_cuda[row_index*COLS2+col_index]=prod;

	}
}

int main(){
	
	//check whether dimensions are valid for a multiplication
	if(COLS1!=ROWS2){
		printf("Matrix dimensions are invalid for matrix multiplication\n");
		exit(1);
	}
	
	//Initialize arrays in heap

	int * matA = (int *)malloc(sizeof(int)*SIZE1);
	int * matB = (int *)malloc(sizeof(int)*SIZE2);
	int * matC = (int *)malloc(sizeof(int)*SIZE3);
	
	//generate some values for matrixA
	int i,j;
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			matA[i*COLS1+j]=i+j;
		}
	}

	//print the matA 
	printf("Matrix A : \n");
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS1;j++){
			printf("%5d ",matA[i*COLS1+j]);
		}
		printf("\n");
	}		
	printf("\n");

	//generate values for matrixB
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			matB[i*COLS2+j]=i-j;
		}
	}

	//print the matB
	printf("Matrix B : \n");
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			printf("%5d ",matB[i*COLS2+j]);
		}
		printf("\n");
	}	
	printf("\n");
	
	//cuda stuff starts here

	//cuda pointers
	int * matA_cuda; int * matB_cuda; int * matC_cuda;

	//allocate GPU memory
	cudaMalloc((void **)&matA_cuda,sizeof(int)*SIZE1);checkCudaError();
	cudaMalloc((void **)&matB_cuda,sizeof(int)*SIZE2);checkCudaError();
	cudaMalloc((void **)&matC_cuda,sizeof(int)*SIZE3);checkCudaError();

	//copy data to GPU memory
	cudaMemcpy(matA_cuda,matA,sizeof(int)*SIZE1,cudaMemcpyHostToDevice);checkCudaError();
	cudaMemcpy(matB_cuda,matB,sizeof(int)*SIZE2,cudaMemcpyHostToDevice);checkCudaError();
	
	dim3 blockSize(BLOCK_SIZE,BLOCK_SIZE);
    dim3 numBlocks(ceil(COLS2/float(BLOCK_SIZE)),ceil(ROWS1/float(BLOCK_SIZE)));

	//multiply the matrices
	matMul<<<numBlocks,blockSize>>>(matC_cuda,matA_cuda,matB_cuda);
	cudaDeviceSynchronize();
	checkCudaError();

	cudaMemcpy(matC,matC_cuda,sizeof(int)*SIZE3,cudaMemcpyDeviceToHost); 
	checkCudaError();



	//print the answer
	printf("Answer : \n");	
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%5d ",matC[i*COLS2+j]);
		}
		printf("\n");
	}
	
	free(matA);free(matB);free(matC);
	cudaFree(matA_cuda);cudaFree(matB_cuda);cudaFree(matC_cuda);
	
	return 0;

}
