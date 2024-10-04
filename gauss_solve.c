#include "gauss_solve.h"
#include <math.h>
#include <stdio.h>  // For error handling

// Function to check if any element satisfies a condition
void plu(int n, double A[n][n], int P[n]) {
    // Initialize the permutation matrix P
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    // Perform LU factorization with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find the pivot row
        int maxIndex = k;
        double maxValue = fabs(A[k][k]);
        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i][k]) > maxValue) {
                maxValue = fabs(A[i][k]);
                maxIndex = i;
            }
        }

        // Swap rows in P and A if necessary
        if (maxIndex != k) {
            int temp = P[k];
            P[k] = P[maxIndex];
            P[maxIndex] = temp;

            for (int j = 0; j < n; j++) {
                double tmp = A[k][j];
                A[k][j] = A[maxIndex][j];
                A[maxIndex][j] = tmp;
            }
        }

        // Perform the LU decomposition
        for (int i = k + 1; i < n; i++) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

void print_matrix(int n, double A[n][n]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", A[i][j]);
        }
        printf("\n");
    }
}

void gauss_solve_in_place(const int n, double A[n][n], double b[n]) {
    for (int k = 0; k < n; ++k) {
        // Use the any function to check for a zero pivot
        if (any(A[k], n, 1e-12)) {
            fprintf(stderr, "Error: Zero pivot encountered at index %d\n", k);
            return; // Return early to avoid further errors
        }

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
        // Use the any function to check for a zero division
        if (any(A[i], n, 1e-12)) {
            fprintf(stderr, "Error: Zero division encountered during back-substitution at index %d\n", i);
            return; // Return early to avoid further errors
        }

        for (int j = i + 1; j < n; ++j) {
            b[i] -= A[i][j] * b[j];
        }
        b[i] /= A[i][i];  // Ensure we do not divide by zero here
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
            
            // Check if A[k][k] is close to zero using the any function
            if (!any(&A[k][k], 1, 1e-12)) {
                A[i][k] /= A[k][k];	
            } else {
                fprintf(stderr, "Error: Zero division in LU decomposition at index %d\n", k);
                return; // Return early to avoid further errors
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
