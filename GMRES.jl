using LinearAlgebra
using HDF5

io = open("errorsGMRES.csv","a")
write(io, "\t index, err\n")

h5open("Ab.h5","r") do file
    A = read(file, "A-Matrix")
    b = read(file, "b-Vector")

    k = 50
    n = size(A,1)
    Q = zeros(n,k+1)
    H = zeros(k+1,k)

    beta = norm(b,2)
    Q[:,1] = b/beta

    for j = 1:k
        z = A*Q[:,j]
        for i = 1:j
            H[i,j] = dot(Q[:,i], z)
            z = z - H[i,j]*Q[:,i]
        end
        z = z - Q[:,1:j]*(Q[:,1:j]'*z)
        H[j+1,j] = norm(z,2)
        H[j+1,j] == 0 ? break : Q[:,j+1] = z/H[j+1,j]
    end

    for t = 1:k
        q,r = qr(H[1:t+1,1:t])
        y = r\q[1,1:end-1]*beta
        x = Q[:,1:t]*y
        error = norm(b - A*x, 2)/beta
        write(io, "\t $t, $error \n")
    end
end

close(io)
