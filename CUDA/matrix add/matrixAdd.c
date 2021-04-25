#include <stdio.h>

#define SIZE 10

void matrixAdd(int matA[SIZE][SIZE],int matB[SIZE][SIZE],int matAns[SIZE][SIZE]);
void printResults(int mat[SIZE][SIZE]);
int main(){
    int matA[SIZE][SIZE];
    int matB[SIZE][SIZE];
    int matAns[SIZE][SIZE];    
    
    //initialize matrixes
    int row=0;
    for(row=0;row<SIZE;row++){
        int col=0;
        for(col=0;col<SIZE;col++){
            matA[row][col]=row;
            matB[row][col]=SIZE-row;
        }
    }
    matrixAdd(matA,matB,matAns);
    printResults(matAns); 
    return 0;
}

void matrixAdd(int matA[SIZE][SIZE],int matB[SIZE][SIZE],int matAns[SIZE][SIZE]){
    int row=0;
    for(row=0;row<SIZE;row++){
        int col=0;
        for(col=0;col<SIZE;col++){
            matAns[row][col]=matA[row][col]+matB[row][col];
        }
    }
}

void printResults(int mat[SIZE][SIZE]){
    int row=0;
    for(row=0;row<SIZE;row++){
        int col=0;
        for(col=0;col<SIZE;col++){
            fprintf(stderr,"%d ",mat[row][col]);
        }
        fprintf(stderr,"\n");
    }
}