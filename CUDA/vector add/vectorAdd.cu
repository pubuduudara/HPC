#include <stdio.h>

#define SIZE 12000
#define BLOCK_SIZE 256

#define checkCudaError(){ gpuAssert(__FILE__, __LINE__);}

void gpuAssert(const char * file,int line){
    cudaError_t code=cudaGetLastError();
    if(code!=cudaSuccess){
        fprintf(stderr,"Cuda error: %s \n in file : %s line number : %d\n",cudaGetErrorString(code),file,line);
        exit(1);
    }
}

__global__ void addVectors(int *arA_cuda, int *arB_cuda, int *answ_cuda);

int main(){
    //initialize cpu data variables
    int arA[SIZE];
    int arB[SIZE];
    int ans[SIZE];

    int i;
    for(i=0;i<SIZE;i++){
        arA[i]=i;
        arB[i]=SIZE-i;
    }

    //create GPU memory pointers

    int *arA_cuda;
    int *arB_cuda;
    int *answ_cuda;

    //latest error code
    cudaError_t code;

    // allocate GPU memeory
    cudaMalloc((void **)&arA_cuda,sizeof(int)*SIZE);
    checkCudaError();
    
    cudaMalloc((void **)&arB_cuda,sizeof(int)*SIZE);
    checkCudaError();


    cudaMalloc((void **)&answ_cuda,sizeof(int)*SIZE);
    checkCudaError();

    //copy data to GPU memory
    cudaMemcpy(arA_cuda,&arA,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    checkCudaError();

    cudaMemcpy(arB_cuda,&arB,sizeof(int)*SIZE,cudaMemcpyHostToDevice);
    checkCudaError();

    //number of blocks
    int no_of_blocks=ceil(SIZE/float(BLOCK_SIZE));

    //execute kernel
    addVectors<<<no_of_blocks,BLOCK_SIZE>>>(arA_cuda,arB_cuda,answ_cuda);
    cudaDeviceSynchronize();
    checkCudaError();

    //copy results back to host
    cudaMemcpy(&ans,answ_cuda,sizeof(int)*SIZE,cudaMemcpyDeviceToHost);
    checkCudaError();

    
    for(i=0;i<SIZE;i++){
        fprintf(stderr,"%d ",ans[i]);
    }
    fprintf(stderr,"\n");

    return 0;
    cudaFree(arA_cuda);
    cudaFree(arB_cuda);
    cudaFree(answ_cuda);
}

__global__ void addVectors(int *arA_cuda, int *arB_cuda, int *answ_cuda){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<SIZE){
        answ_cuda[tid]=arA_cuda[tid]+ arB_cuda[tid];
    }

}