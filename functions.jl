using LinearAlgebra
using ImageView
using HDF5

function Centroid(Training::String, Testing::String)
    # Training and Testing must be names of h5 files
    # containing datasets "key" and "data"
    average = zeros(10,256)
    avg_count = zeros(10,1)
    h5open(Training, "r") do file
        key = read(file, "key")
        data = read(file, "data")
        for i = 1:length(key)
            k = convert(Int64,key[i]) + 1
            avg_count[k] = avg_count[k] + 1
            average[k,:] = average[k,:] + data[256*i-255:256*i]
        end
    end
    [average[i,:] = average[i,:]/avg_count[i] for i = 1:10]

    status = zeros(10,2)
    h5open(Testing, "r") do file
        key = read(file, "key")
        data = read(file, "data")
        for i = 1:length(key)
            dataImg = data[256*i-255:256*i]
            k = argmin([norm(dataImg - average[i,:],2) for i = 1:10])
            status = check(k, key[i]+1, status)
        end
    end
    return results(status)
end

function svdMethod(Training::String, Testing::String)
    # Training and Testing must be names of h5 files
    # containing datasets "key" and "data"
    SValues = 9:-2:1
    svdU = zeros(10, 256, SValues[1])
    h5open(Training, "r") do file
        key = read(file, "key") # range in 0:9
        data = reshape(read(file, "data"), (256,1707))
        for i = 1:10
            I = findall(x->x==i-1, key[:])
            U,_,_ = svd(data[:,I])
            svdU[i,:,:] = U[:,1:SValues[1]]
        end
    end

    status = zeros(5,10,2)
    h5open(Testing, "r") do file
        key = read(file, "key")
        data = reshape(read(file, "data"), (256,2007))
        for h = 1:length(SValues)
            sv = SValues[h]
            for j = 1:size(data,2)
                x = data[:,j]
                J = 1:sv
                dist = [norm(x - svdU[i,:,J]*(svdU[i,:,J]'*x)) for i = 1:10]
                status[h,:,:] = check(argmin(dist), key[j]+1, status[h,:,:])
            end
        end
    end
    return [results(status[h,:,:]) for h in 1:5]
end

function check(min, key, status)
    key = convert(Int64,key)
    if min == key
        status[key, 1] += 1
    else
        status[key, 2] += 1
    end
    return status
end

function results(status)
    return [status[i,1]/sum(status[i,:]) for i = 1:10]
end
