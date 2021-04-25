#include <stdio.h>

#define ROWS 10
#define COLS 10
#define SIZE ROWS*COLS
#define BLOCK_SIZE 16

//cuda error check fuctnion
#define checkCudaError(){ gpuAssert(__FILE__, __LINE__);}
void gpuAssert(const char * file,int line){
    cudaError_t code=cudaGetLastError();
    if(code!=cudaSuccess){
        fprintf(stderr,"Cuda error: %s \n in file : %s line number : %d\n",cudaGetErrorString(code),file,line);
        exit(1);
    }
}


__global__ void matrixAdd(int *matA,int *matB,int *matAns);
void printResults(int mat[ROWS][COLS]);

int main(){
    int matA[ROWS][COLS];
    int matB[ROWS][COLS];
    int matAns[ROWS][COLS];    
    
    //initialize matrixes
    int row=0;
    for(row=0;row<ROWS;row++){
        int col=0;
        for(col=0;col<COLS;col++){
            matA[row][col]=row;
            matB[row][col]=SIZE-row;
        }
    }
    
    //cuda initializations
    int *matA_cuda;int *matB_cuda;int *matAns_cuda;
    //cuda memory allocation
    cudaMalloc((void **)&matA_cuda,sizeof(int)*SIZE);
    checkCudaError();
    cudaMalloc((void **)&matB_cuda,sizeof(int)*SIZE);
    checkCudaError();
    cudaMalloc((void **)&matAns_cuda,sizeof(int)*SIZE);
    checkCudaError();

    cudaMemcpy(matA_cuda,matA,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    checkCudaError();
    cudaMemcpy(matB_cuda,matB,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    checkCudaError();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numberBlocks(ceil(COLS/float(BLOCK_SIZE)),ceil(ROWS/float(BLOCK_SIZE)));

    matrixAdd<<<numberBlocks,threadsPerBlock>>>(matA_cuda,matB_cuda,matAns_cuda); 
    cudaDeviceSynchronize();
    checkCudaError();

    cudaMemcpy(matAns,matAns_cuda,sizeof(int)*SIZE,cudaMemcpyDeviceToHost);
    checkCudaError();

    cudaFree(matA_cuda);
    cudaFree(matB_cuda);
    cudaFree(matAns_cuda);
    printResults(matAns);

    return 0;
}

__global__ void matrixAdd(int *matA,int *matB,int *matAns){
    int row_index=blockDim.y*blockIdx.y+threadIdx.y;
    int col_index=blockDim.x*blockIdx.x+threadIdx.x; 
    

    if(row_index<ROWS && col_index<COLS){
        int position=row_index*COLS+ col_index;
        matAns[position]=matA[position]+matB[position]; 
    }

}

void printResults(int mat[ROWS][COLS]){
    int row=0;
    for(row=0;row<ROWS;row++){
        int col=0;
        for(col=0;col<COLS;col++){
            fprintf(stderr,"%d ",mat[row][col]);
        }
        fprintf(stderr,"\n");
    }
}