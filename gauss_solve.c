#include "gauss_solve.h"
#include <math.h>
#include <stdio.h>  // For error handling

// Function to check if any element satisfies a condition
int any(double *arr, int n, double threshold) {
    for (int i = 0; i < n; i++) {
        if (fabs(arr[i]) < threshold) {
            return 1;  // Found an element that satisfies the condition
        }
    }
    return 0;  // No element satisfied the condition
}

// Function to check if all elements satisfy a condition
int all(double *arr, int n, double threshold) {
    for (int i = 0; i < n; i++) {
        if (fabs(arr[i]) >= threshold) {
            return 0;  // Found an element that does not satisfy the condition
        }
    }
    return 1;  // All elements satisfied the condition
}

void swap_rows(double A[], int n, int row1, int row2) {
    for (int i = 0; i < n; i++) {
        double temp = A[row1 * n + i];
        A[row1 * n + i] = A[row2 * n + i];
        A[row2 * n + i] = temp;
    }
}

void plu(int n, double A[n][n], int P[n]) {
    // Initialize the permutation array P as identity
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    // Perform the PLU decomposition
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
            swap_rows((double *)A, n, k, maxIndex);

            // Swap the corresponding entries in P
            int temp = P[k];
            P[k] = P[maxIndex];
            P[maxIndex] = temp;
        }

        if (fabs(A[k][k]) < 1e-12) {
            fprintf(stderr, "Error: Zero pivot encountered at index %d\n", k);
            return; 
        }

        // Decompose into L and U
        for (int i = k + 1; i < n; i++) {
            // Use the current pivot value for the division
            A[i][k] /= A[k][k];  // Compute L[i][k]
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= A[i][k] * A[k][j];  // Update U[i][j]
            }
        }
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
