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

gauss_library_path = './libgauss.so'

import ctypes
import numpy as np

# Load the shared library (e.g., gauss_solve.so)
lib = ctypes.CDLL('./gauss_solve.so')

# Function signature in the shared C library
lib.plu.argtypes = [ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), 
                    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')]

def plu(A, use_c=False):
    n = len(A)
    A = np.array(A, dtype=np.float64)

    if use_c:
        # C implementation
        P = np.arange(n, dtype=np.int32)
        lib.plu(n, A, P)
        L = np.tril(A, -1) + np.eye(n)
        U = np.triu(A)
        return P.tolist(), L.tolist(), U.tolist()
    else:
        # Python implementation
        P = list(range(n))
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]
        
        for k in range(n):
            # Find pivot
            pivot = max(range(k, n), key=lambda i: abs(A[i][k]))
            if pivot != k:
                A[k], A[pivot] = A[pivot], A[k]
                P[k], P[pivot] = P[pivot], P[k]

            for i in range(k+1, n):
                A[i][k] /= A[k][k]
                for j in range(k+1, n):
                    A[i][j] -= A[i][k] * A[k][j]

        for i in range(n):
            for j in range(n):
                if i > j:
                    L[i][j] = A[i][j]
                elif i == j:
                    L[i][j] = 1.0
                else:
                    U[i][j] = A[i][j]

        return P, L, U

# Test usage
A = [[2.0, 3.0, -1.0],
     [4.0, 1.0, 2.0],
     [-2.0, 7.0, 2.0]]

use_c = False
P, L, U = plu(A, use_c=use_c)
print("Python PLU:")
print("P:", P)
print("L:", L)
print("U:", U)

use_c = True
P, L, U = plu(A, use_c=use_c)
print("C PLU:")
print("P:", P)
print("L:", L)
print("U:", U)


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

use_c = True;
P, L, U = plu(A, use_c = use_c)
