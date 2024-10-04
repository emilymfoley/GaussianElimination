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
import numpy as np

gauss_library_path = './libgauss.so'

def plu(A, use_c=True):
    """
    Perform PLU decomposition (with partial pivoting).
    
    Args:
        A (list of lists or 2D numpy array): The matrix to decompose.
    
    Returns:
        P (2D numpy array): The permutation matrix.
        L (2D numpy array): The lower triangular matrix.
        U (2D numpy array): The upper triangular matrix.
    """
    n = len(A)
    A = np.array(A, dtype=float)
    
    # Create identity matrix for P
    P = np.eye(n)
    # Create zero matrices for L and U
    L = np.zeros((n, n))
    U = np.array(A, dtype=float)  # Initialize U as a copy of A
    
    for k in range(n):
        # Partial pivoting: find the index of the row with the largest absolute value in column k
        pivot = np.argmax(np.abs(U[k:n, k])) + k
        
        # Swap rows in U and P accordingly
        U[[k, pivot]] = U[[pivot, k]]
        P[[k, pivot]] = P[[pivot, k]]
        
        # Also swap the rows of L up to column k
        if k > 0:
            L[[k, pivot], :k] = L[[pivot, k], :k]
        
        # Set the diagonal of L to 1
        L[k, k] = 1
        
        # Perform the elimination below the pivot row
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor  # Store the factor in L
            U[i, k:] -= factor * U[k, k:]  # Ensure proper broadcasting
            U[i, k] = 0  # Explicitly set the element to 0 after updating
    
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

    # Modify the array in C
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
        for i in range(k, n):
            for j in range(k):
                A[i][j] -= A[k][j] * A[i][k]  # Corrected the operation here
        for i in range(k + 1, n):
            for j in range(k):
                A[i][k] -= A[i][j] * A[j][k]
            A[i][k] /= A[k][k]

    return unpack(A)

def lu(A, use_c=False):
    if use_c:
        return lu_c(A)
    else:
        return lu_python(A)

if __name__ == "__main__":
    def get_A():
        """ Make a test matrix """
        A = [[2.0, 3.0, -1.0],
             [4.0, 1.0, 2.0],
             [-2.0, 7.0, 2.0]]
        return A

    A = get_A()
    
    L, U = lu(A, use_c=False)
    print(L)
    print(U)

    # Must re-initialize A as it was destroyed
    A = get_A()

    L, U = lu(A, use_c=True)
    print(L)
    print(U)
