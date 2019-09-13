using LinearAlgebra

function GE( A )
    n,_= size(A) # dimension of A
    B = copy(A)  # Deep copy of A
    # Main Gauss Elimination Alg
    for i = 1:n-1
        if A[i,i] == 0
            return "Failed... Pivot is 0"
        else
            A[i+1:n,i] = A[i+1:n,i]/A[i,i]
            A[i+1:n,i+1:n] = A[i+1:n,i+1:n] - A[i+1:n,i]*A[i,i+1:n]'
        end
    end
    # Setting up U,L from A and evaluating ϵ
    U = zeros(size(A)); L = zeros(size(A));
    [U[i,i:n] = A[i,i:n] for i = 1:n]
    [L[i,1:i-1] = A[i,1:i-1] for i in 2:n]
    [L[i,i] = 1 for i in 1:n]
    err = norm(B-L*U,1)/norm(B,1)
    return L, U, err
end

function GEPP( A::Array{Float64,2} )
    n,_= size(A) # dimension of A
    B = copy(A)  # Deep copy of A
    p = [x for x in 1:n] # permutation index vector
    # Main Gauss Elimination with Partial Pivoting Alg
    for i in 1:n-1
        _,k = findmax([abs(x) for x in A[p[i:n],i]]); k = k+i-1
        length(k) > 1 ? k = k[1] : false
        # k is the smallest index of max abs value in column i
        A[p[k],i] == 0 ? (return "No Unique Solution") : false
        if p[i] != p[k]
            temp = p[i]
            p[i] = p[k]
            p[k] = temp
        end
        k = p[i+1:n]
        A[k,i] = A[k,i]/A[p[i],i]
        A[k,i+1:n] = A[k,i+1:n] - A[k,i]*A[p[i],i+1:n]'
    end
    # Setting up U,L from A and evaluating ϵ
    U = zeros(size(A)); L = zeros(size(A));
    [U[i,i:n] = A[p[i],i:n] for i = 1:n]
    [L[i,1:i-1] = A[p[i],1:i-1] for i in 2:n]
    [L[i,i] = 1 for i in 1:n]
    err = norm(B[p,:]-L*U,1)/norm(B,1)
    return L, U, p, err
end
