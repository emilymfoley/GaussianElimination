/*----------------------------------------------------------------
* File:     gauss_solve.c
*----------------------------------------------------------------
*
* Author:   Marek Rychlik (rychlik@arizona.edu)
* Date:     Sun Sep 22 15:40:29 2024
* Copying:  (C) Marek Rychlik, 2020. All rights reserved.
*
*----------------------------------------------------------------*/
#include "gauss_solve.h"
#include <math.h>
#include <stdio.h>  // For error handling

#include<stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

double plu(int n, double **c, int *p, double tol) {
  int i, j, k, pivot_ind = 0, temp_ind;
  int ii, jj;
  double *vv=calloc(n,sizeof(double));
  double pivot, *temp_row;
  double temp;

  for (j = 0; j < n; ++j) {
    pivot = 0;
    for (i = j; i < n; ++i)
      if (fabs(c[i][j]) > fabs(pivot)) {
        pivot = c[i][j];
        pivot_ind = i;
      }

    temp_row = c[j];
    c[j] = c[pivot_ind];
    c[pivot_ind] = temp_row;

    temp_ind  = p[j];
    p[j] = p[pivot_ind];
    p[pivot_ind] = temp_ind;

    for (k = j+1; k < n; ++k) {
      temp=c[k][j]/=c[j][j];
      for(int q=j+1;q<n;q++){
            c[k][q] -= temp*c[j][q];
          }
    }
    for(int q=0;q<n;q++){
      for(int l=0;l<n;l++){
        printf("%lf ",c[q][l]);
      }
      printf("\n");
    }

  }
  return 0.;
}

int main() {
  double **x;
  x=calloc(3,sizeof(double));
  for(int i=0;i<3;i++){
    x[i]=calloc(3,sizeof(double));
  }
  memcpy(x[0],(double[]){1,2,3},3*sizeof(double));
  memcpy(x[1],(double[]){4,5,6},3*sizeof(double));
  memcpy(x[2],(double[]){7,8,9},3*sizeof(double));

  int *p=calloc(3,sizeof(int));
  memcpy(p,(int[]){0,1,2},3*sizeof(int));
  plupmc(3,x,p,1);
  for(int i=0;i<3;i++){
    free(x[i]);
  }
  free(p);
  free(x);


}

void gauss_solve_in_place(const int n, double A[n][n], double b[n]) {
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            /* Store the multiplier into A[i][k] as it would become 0 and be useless */
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
            b[i] -= A[i][k] * b[k];
        }
    } /* End of Gaussian elimination, start back-substitution. */
    
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            b[i] -= A[i][j] * b[j];
        }
        b[i] /= A[i][i];
    } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[n][n]) {
    for (int k = 0; k < n; ++k) {
        for (int i = k; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                /* U[k][i] -= L[k][j] * U[j][i] */
                A[k][i] -= A[k][j] * A[j][i]; 
            }
        }
        for (int i = k + 1; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                /* L[i][k] -= A[i][k] * U[j][k] */
                A[i][k] -= A[i][j] * A[j][k]; 
            }
            /* L[k][k] /= U[k][k] */
            if (A[k][k] != 0) {
                A[i][k] /= A[k][k];	
            }
        }
    }
}

void lu_in_place_reconstruct(int n, double A[n][n]) {
    for (int k = n - 1; k >= 0; --k) {
        for (int i = k + 1; i < n; ++i) {
            A[i][k] *= A[k][k];
            for (int j = 0; j < k; ++j) {
                A[i][k] += A[i][j] * A[j][k];
            }
        }
        for (int i = k; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                A[k][i] += A[k][j] * A[j][i];
            }
        }
    }
}
