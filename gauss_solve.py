#----------------------------------------------------------------
# File:     gauss_solve.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Thu Sep 26 10:38:32 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A Python wrapper module around the C library libgauss.so

import ctypes

lib = './libgauss.so'

import numpy as np

def plu_python(A, use_c=False):
    """
    Perform PLU decomposition (with partial pivoting).
    
    Args:
        A (list of lists): The matrix to decompose.
        use_c (bool): If True, use the C implementation (currently ignored).
    
    Returns:
        P (list of integers): The permutation list.
        L (list of lists): The lower triangular matrix.
        U (list of lists): The upper triangular matrix.
    """
    n = len(A)

    # Initialize P as a list representing row swaps (identity initially)
    P = list(range(n))
    
    # Initialize L as a zero matrix and U as a copy of A
    L = [[0.0] * n for _ in range(n)]
    U = [row[:] for row in A]  # Copy of A to avoid modifying the original matrix
    
    for k in range(n):
        # Partial pivoting: find the index of the row with the largest absolute value in column k
        pivot = max(range(k, n), key=lambda i: abs(U[i][k]))
        
        # Swap rows in U and P accordingly
        U[k], U[pivot] = U[pivot], U[k]
        P[k], P[pivot] = P[pivot], P[k]
        
        # Also swap the rows of L up to column k
        if k > 0:
            L[k][:k], L[pivot][:k] = L[pivot][:k], L[k][:k]
        
        # Set the diagonal of L to 1
        L[k][k] = 1
        
        # Perform the elimination below the pivot row
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor  # Store the factor in L
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
    
    return P, L, U

def plu_c(A):
    """
    Call the C implementation of PLU decomposition.
    
    Args:
        A (list of lists): The matrix to decompose.

    Returns:
        P (list): The permutation vector.
        L (list of lists): The lower triangular matrix.
        U (list of lists): The upper triangular matrix.
    """
    n = len(A)
    # Create an array of double for A and an integer array for P
    A_c = (ctypes.c_double * (n * n))(*[A[i][j] for i in range(n) for j in range(n)])  # Flattened A
    P_c = (ctypes.c_int * n)()  # Create an integer array for P

    # Call the C function
    lib.plu(n, A_c, P_c)

    # Convert the results back to Python lists
    # Convert A_c back to a 2D list for L and U
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    # Fill L and U from A_c
    for i in range(n):
        for j in range(n):
            if i < j:
                L[i][j] = 0.0
            elif i == j:
                L[i][j] = 1.0
            else:
                L[i][j] = A_c[i * n + j]  # Access flattened array
            U[i][j] = A_c[i * n + j]  # Access flattened array for U

    # Convert P_c to a Python list
    P = [P_c[i] for i in range(n)]

    return P, L, U


def unpack(A):
    """ Extract L and U parts from A, fill with 0's and 1's """
    n = len(A)
    L = [[A[i][j] for j in range(i)] + [1] + [0 for j in range(i+1, n)]
         for i in range(n)]

    U = [[0 for j in range(i)] + [A[i][j] for j in range(i, n)]
         for i in range(n)]

    return L, U

def lu_c(A):
    """ Accepts a list of lists A of floats and
    it returns (L, U) - the LU-decomposition as a tuple.
    """
    # Load the shared library
    lib = ctypes.CDLL(gauss_library_path)

    # Create a 2D array in Python and flatten it
    n = len(A)
    flat_array_2d = [item for row in A for item in row]

    # Convert to a ctypes array
    c_array_2d = (ctypes.c_double * len(flat_array_2d))(*flat_array_2d)

    # Define the function signature
    lib.lu_in_place.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    # Modify the array in C (e.g., add 10 to each element)
    lib.lu_in_place(n, c_array_2d)

    # Convert back to a 2D Python list of lists
    modified_array_2d = [
        [c_array_2d[i * n + j] for j in range(n)]
        for i in range(n)
    ]

    # Extract L and U parts from A, fill with 0's and 1's
    return unpack(modified_array_2d)

def lu_python(A):
    n = len(A)
    for k in range(n):
        for i in range(k,n):
            for j in range(k):
                A[k][i] -= A[k][j] * A[j][i]
        for i in range(k+1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)


def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)
        
def plu(A, use_c=False):
    if use_c:
        return plu_c(A)
    else:
        return plu_python(A)



if __name__ == "__main__":

    def get_A():
        """ Make a test matrix """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    A = get_A()

    L, U = lu(A, use_c = False)
    print(L)
    print(U)

    # Must re-initialize A as it was destroyed
    A = get_A()

    L, U = lu(A, use_c=True)
    print(L)
    print(U)
