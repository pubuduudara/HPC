
#define checkCudaError(){gpuAssert(__FILE__,__LINE__);}

static inline void gpuAssert(const char *file, int line){
    cudaError_t ret= cudaGetLastError();
    if(ret != cudaSuccess){
        fprintf(stderr,"Error %s at %s at line %d",cudaGetErrorString(ret),file,line);
        exit(1);
    }
}