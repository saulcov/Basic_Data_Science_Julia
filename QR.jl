using LinearAlgebra

function errorsQR(A, Q, R; p = 1 )
     return norm(A - Q*R ,p)/norm(R ,p)
end

function CGS( A ) # O(2mn^2)
    m,n = size(A)
    R = zeros(n,n)
    Q = zeros(m,n)
    for i = 1:n
        h = A[:,i]
        for j = 1:i-1 # (4m-1)(i-1) flops
            R[j,i] = dot(Q[:,j],A[:,i]) # (2m-1) flops
            h = h - R[j,i]*Q[:,j] #2m flops
        end
        R[i,i] = norm(h,2) # (2m-1) flops
        Q[:,i] = h/R[i,i] # m flops
    end
    return Q, R, errorsQR(A, Q, R)
end

function MGS( A )
    m,n = size(A)
    R = zeros(n,n)
    Q = zeros(m,n)
    for i = 1:n
        h = A[:,i]
        for j = 1:i-1
            R[j,i] = dot(Q[:,j],h)
            h = h - R[j,i]*Q[:,j]
        end
        R[i,i] = norm(h,2)
        Q[:,i] = h/R[i,i]
    end
    return Q, R, errorsQR(A, Q, R)
end
