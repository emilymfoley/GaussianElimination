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

void swap_rows(double A[], int n, int row1, int row2) {
    for (int i = 0; i < n; i++) {
        double temp = A[row1 * n + i];
        A[row1 * n + i] = A[row2 * n + i];
        A[row2 * n + i] = temp;
    }
}

void plu(int n, double A[], int P[]) {  // Use A[] as 1D array
    // Initialize the permutation array P as identity
    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    // Perform the PLU decomposition
    for (int k = 0; k < n; k++) {
        // Find the pivot (largest absolute value in the current column)
        int maxIndex = k;
        double maxVal = fabs(A[k * n + k]);

        for (int i = k + 1; i < n; i++) {
            if (fabs(A[i * n + k]) > maxVal) {
                maxVal = fabs(A[i * n + k]);
                maxIndex = i;
            }
        }

        // Swap rows if necessary
        if (maxIndex != k) {
            swap_rows(A, n, k, maxIndex);

            // Swap the corresponding entries in P
            int temp = P[k];
            P[k] = P[maxIndex];
            P[maxIndex] = temp;
        }

        // Check for zero pivot element to avoid division by zero
        if (fabs(A[k * n + k]) < 1e-12) {
            fprintf(stderr, "Error: Zero pivot encountered at index %d\n", k);
            return; // Return early to avoid further errors
        }

        // Decompose into L and U
        for (int i = k + 1; i < n; i++) {
            A[i * n + k] /= A[k * n + k];  // Fixed indexing
            for (int j = k + 1; j < n; j++) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];  // Fixed indexing
            }
        }
    }
}

void gauss_solve_in_place(const int n, double A[], double b[]) {
    for (int k = 0; k < n; ++k) {
        for (int i = k + 1; i < n; ++i) {
            /* Store the multiplier into A[i][k] as it would become 0 and be useless */
            A[i * n + k] /= A[k * n + k];  // Fixed indexing
            for (int j = k + 1; j < n; ++j) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];  // Fixed indexing
            }
            b[i] -= A[i * n + k] * b[k];  // Fixed indexing
        }
    } /* End of Gaussian elimination, start back-substitution. */
    
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            b[i] -= A[i * n + j] * b[j];  // Fixed indexing
        }
        b[i] /= A[i * n + i];  // Fixed indexing
    } /* End of back-substitution. */
}

void lu_in_place(const int n, double A[]) {
    for (int k = 0; k < n; ++k) {
        for (int i = k; i < n; ++i) {
            for (int j = 0; j < k;
