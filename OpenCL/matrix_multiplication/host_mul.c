#include <stdio.h>
//OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "error.h"

#include "mul.h"





int main(){

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("kernel_mul.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    //--------------initializations-------------------------------

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms); checkError(ret, "Failed clGetPlatformIDs");
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices); checkError(ret, "Failed clGetDeviceIDs");

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret); checkError(ret, "Failed clCreateContext");

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);checkError(ret, "Failed clCreatclCreateCommandQueueeContext");

    //------------------coding-------------------------------------------------

    //host data
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

    //generate values for matrixB
	for(i=0;i<ROWS2;i++){
		for(j=0;j<COLS2;j++){
			matB[i*COLS2+j]=i-j;
		}
	}

    //device data
    cl_mem matA_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE1 * sizeof(int), NULL, &ret);checkError(ret, "Failed clCreateBuffer matA_cl");
    cl_mem matB_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE2 * sizeof(int), NULL, &ret);checkError(ret, "Failed clCreateBuffer matB_cl");
    cl_mem matC_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE3 * sizeof(int), NULL, &ret);checkError(ret, "Failed clCreateBuffer matC_cl");

    //copy data to GPU memory
    ret = clEnqueueWriteBuffer(command_queue, matA_cl, CL_TRUE, 0, SIZE1 * sizeof(int), matA, 0, NULL, NULL);checkError(ret, "Failed clEnqueueWriteBuffer matA_cl");
    ret = clEnqueueWriteBuffer(command_queue, matB_cl, CL_TRUE, 0, SIZE2 * sizeof(int), matB, 0, NULL, NULL);checkError(ret, "Failed clEnqueueWriteBuffer matB_cl");

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);checkError(ret, "Failed clCreateProgramWithSource");

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL); checkError(ret, "Failed clBuildProgram");

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matMul", &ret); checkError(ret, "Failed clCreateKernel");

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&matA_cl); checkError(ret, "Failed clSetKernelArg matA_cl");
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matB_cl); checkError(ret, "Failed clSetKernelArg matB_cl");
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&matC_cl); checkError(ret, "Failed clSetKernelArg matC_cl");

    // Execute the OpenCL kernel on the list
    // size_t global_item_size = SIZE3; // Process the entire lists
    // size_t local_item_size = 64; // Process in groups of 64


    const size_t global_item_size[2] = {ROWS1, COLS2}; //global
    const size_t local_item_size[2] = {BLOCK_SIZE,BLOCK_SIZE};//local

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL); checkError(ret, "Failed clEnqueueNDRangeKernel");
    ret = clFinish(command_queue); checkError(ret, "Failed clFinish");

    //data transfer device to host
    ret = clEnqueueReadBuffer(command_queue, matC_cl, CL_TRUE, 0, SIZE3 * sizeof(int), matC, 0, NULL, NULL); checkError(ret, "Failed clEnqueueReadBuffer");


    //print the answer
	printf("Answer : \n");	
	for(i=0;i<ROWS1;i++){
		for(j=0;j<COLS2;j++){
			printf("%5d ",matC[i*COLS2+j]);
		}
		printf("\n");
	}

    //free allocations

    ret = clReleaseKernel(kernel); checkError(ret, "Failed clReleaseKernel");
    ret = clReleaseProgram(program); checkError(ret, "Failed clReleaseProgram");
    ret = clReleaseMemObject(matA_cl); checkError(ret, "Failed clReleaseMemObject");
    ret = clReleaseMemObject(matB_cl); checkError(ret, "Failed clReleaseMemObject");
    ret = clReleaseMemObject(matC_cl); checkError(ret, "Failed clReleaseMemObject");
    ret = clReleaseCommandQueue(command_queue); checkError(ret, "Failed clReleaseCommandQueue");
    ret = clReleaseContext(context); checkError(ret, "Failed context");
    free(matA);
    free(matB);
    free(matC);






    return 0;
}