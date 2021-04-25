#include "mul.h"
__kernel void matMul(__global int * matA_cl, __global int * matB_cl, __global int * matC_cl) {
    
    //int row_index=blockDim.y*blockIdx.y+threadIdx.y;
    int row_index=get_global_id(1);

    //int col_index=blockDim.x*blockIdx.x+threadIdx.x; 
    int col_index=get_global_id(0);

	if(row_index<ROWS1 && col_index<COLS2){
		int prod=0;
		int k;
		for(k=0;k<COLS1;k++){
			prod=prod+matA_cl[row_index*COLS1+k]*matB_cl[k*COLS2+col_index];
		}
		matC_cl[row_index*COLS2+col_index]=prod;

	}
}