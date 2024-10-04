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
            if U[k, k] == 0:
                raise ValueError("Zero pivot encountered.")

            factor = U[i, k] / U[k, k]  # Calculate the factor for elimination
            L[i, k] = factor  # Store the factor in L
            
            # Ensure correct broadcasting during row operation
            U[i, k:] = U[i, k:] - factor * U[k, k:]

    return P, L, U
