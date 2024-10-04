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

#include <stdio.h>
#include <stdlib.h>  // For exit function
#include <math.h>    // For fabs function

void plu(int n, double A[n][n], int P[n]) {
    // Initialize the permutation array P as identity
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    // Perform the PLU decomposition with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find the pivot (largest absolute value in the current column)
        int maxIndex = k;
        double maxVal = fabs(A[k][k]);

        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i][k]) > maxVal) {
                maxVal = fabs(A[i][k]);
                maxIndex = i;
            }
        }

        // Swap rows if necessary
        if (maxIndex != k) {
            // Swap the rows in A
            for (int j = 0; j < n; j++) {
                double temp = A[k][j];
                A[k][j] = A[maxIndex][j];
                A[maxIndex][j] = temp;
            }

            // Swap the corresponding entries in P
            int temp = P[k];
            P[k] = P[maxIndex];
            P[maxIndex] = temp;
        }

        // Check for zero pivot element to avoid division by zero
        if (fabs(A[k][k]) < 1e-12) {
            fprintf(stderr, "Error: Zero pivot encountered at index %d\n", k);
            exit(EXIT_FAILURE); // Exit the program if zero pivot is found
        }

        // Decompose into L and U
        for (int i = k + 1; i < n; i++) {
            A[i][k] /= A[k][k];  // Compute L[i][k]
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];  // Update U[i][j]
            }
        }
    }
}

int main() {
    double A[20][20] = {0}; // Initialize the matrix
    double B[20] = {0}, X[20] = {0}, Y[20] = {0};
    int P[20] = {0}; // Permutation array
    int i, j, n;

    // Read the order of the square matrix
    scanf("%d", &n);

    // Read matrix elements
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            scanf("%lf", &A[i][j]);
        }
    }

    // Read the constant terms
    for (i = 0; i < n; i++) {
        scanf("%lf", &B[i]);
    }

    // Perform LU decomposition with partial pivoting
    plu(n, A, P);

    // Forward substitution to solve for Y
    for (i = 0; i < n; i++) {
        Y[i] = B[P[i]]; // Use the permutation array
        for (j = 0; j < i; j++) {
            Y[i] -= A[i][j] * Y[j];
        }
    }

    // Back substitution to solve for X
    for (i = n - 1; i >= 0; i--) {
        X[i] = Y[i];
        for (j = i + 1; j < n; j++) {
            X[i] -= A[i][j] * X[j];
        }
        X[i] /= A[i][i]; // Avoid division by zero
    }

    return 0; // Standard return for main
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
