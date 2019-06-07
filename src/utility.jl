using LinearAlgebra
using Random

function collapseDim(inp::Array{T}, dim::Int64=1) where T <: Real
    dropdims(sum(inp, dims=dim), dims=dim)
end
function collapseDim(inp::Array{Complex{T}}, dim::Int64=1) where T <: Real
    dropdims(sum(inp, dims=dim), dims=dim)
end

function riemannSum(inp, x, dim::Int64=1)
    dx = (x[end] - x[1])/size(x)[1]
    dropdims(sum(inp, dims=dim), dims=dim).*dx
end

function conv_FortranStrToComplex(inp::AbstractString)
    inp_list = parse.(Float64, split(inp))
    if length(inp_list) == 0
        throw(ArgumentError("Invalid string"))
    end
    real_part= inp_list[1:2:end]
    imag_part= inp_list[2:2:end]
    if length(real_part) != length(imag_part)
        throw(ArgumentError("Not an even number of floats provided"))
    end
    res = [Complex{Float64}(real_part[i],imag_part[i]) for i in 1:length(real_part)]
end


function compare3d(Fₖ1, Fₖ2)
    A = Fₖ1.f_grid
    B = Fₖ2.f_grid
    for i in eachindex(A)
        if !isapprox(A[i] , B[i], atol=1e-15)
            println("Comparison failed at: ")
            println(i)
            println(A[i])
            println(B[i])
            return false
        end
    end
    true
end


# === Generate k-Path from full grid ===

function min_diff(x::Array{Float64})
    res = 2*maximum(abs.(x))
    for i = 1:(length(x)-1)
        dx = abs(x[i] - x[i+1])
        if dx < res && (dx > 0)
            res = dx
        end
    end
    return res
end

# TODO: only works for even spacings along a dimension
function k_spacings(H)
    if (size(H.k_grid)[2] != 3)
        throw(BoundsError("k Grid not over 3 dimensions"))
    end
    kg = H.k_grid
    dx = min_diff(kg[:,1])
    dy = min_diff(kg[:,2])
    dz = min_diff(kg[:,3])
    return dx, dy, dz
end

function Nk(H)
    Nx = length(unique(H.k_grid[:,1]))
    Ny = length(unique(H.k_grid[:,2]))
    Nz = length(unique(H.k_grid[:,3]))
    return Nx, Ny, Nz
end

H_of_k_util(H::F_k, ind::Int64) = H.f_grid[ind,:,:]
H_of_k_util(H::F_kω, ind::Int64) = H.f_grid[ind,:,:,:]

function H_of_k(H, k::Array{<:Real})
    if !all(size(k) .== (3,))
        throw(ArgumentError("Wrong Dimension for k"))
    end
    ind = 1
    #TODO: assumes fortran ordering for now
    diff = H.k_grid[2,3] - H.k_grid[1,3]
    N = round(Int, (size(H.k_grid)[1])^(1/3))
    ind = round(Int, sum((k ./ diff) .* [N*N, N, 1])) + 1
    #if !all(isapprox(k, H.k_grid[ind,:], atol=0.0001))
    ##    print("Loss of precision warning! found ")
    #    print(H.k_grid[ind,:])
    #    print(" instead of ")
    #    println(k)
    #end
    #while ind < size(H.k_grid)[1]+1
    #    if isapprox(H.k_grid[ind,:], k)
    #        break
    #    end
    #    ind += 1
    #end
    if ind > size(H.k_grid)[1]
        print("could not find")
        println(k)
        throw(BoundsError("Item not in array"))
    end
    return H_of_k_util(H, ind)
end

function gen_kGrid(H, k_path::Array{Float64})
    dk = collect(k_spacings(H))
    point = k_path[1,:]
    new_k_grid = [point[1] point[2] point[3]]
    current_k = copy(point)
    for i = 2:(size(k_path)[1])
        next_point = k_path[i,:]
        diff = next_point .- point
        direction = 1.0.*(diff .> 0) - 1.0.*(diff .< 0)
        while true
            current_k = current_k .+ (dk .* direction)  
            new_k_grid = vcat(new_k_grid, transpose(current_k))
            if any(direction .* current_k .> direction .* next_point) ||
                all(direction .* current_k .== direction .* next_point) 
                break
            end
        end
        point = next_point
    end
    return new_k_grid
end

function gen_kPath(H::F_kω, k_path::Array{Float64})
    println("generating kGrid")
    new_k_grid = gen_kGrid(H, k_path)
    println("generating band Hamiltonian")
    size_h = size(H.f_grid)
    new_H_grid = Array{typeof(H.f_grid[1,1,1])}(undef, size(new_k_grid)[1], size_h[2], size_h[3], size_h[4])
    for i = 1:size(new_k_grid)[1]
        new_H_grid[i,:,:,:] = H_of_k(H, new_k_grid[i,:])
    end
    res::F_kω = F_kω(new_k_grid, H.ω_grid, new_H_grid)
    return res
end

function gen_kPath(H::F_k, k_path::Array{Float64})
    println("generating kGrid")
    new_k_grid = gen_kGrid(H, k_path)
    println("generating band Hamiltonian")
    new_H_grid = Array{typeof(H.f_grid[1,1,1])}(undef, size(new_k_grid)[1], size(H.f_grid[1,:,:])[1], size(H.f_grid[1,:,:])[2])
    for i = 1:size(new_k_grid)[1]
        new_H_grid[i,:,:] = H_of_k(H, new_k_grid[i,:])
    end
    res::F_k = F_k(new_k_grid, new_H_grid)
    return res
end

randomU(rng, size) = qr(randn(rng, Complex{Float64}, size)).Q

function rotateCoulomb(Uijkl, Utf)
    n_orb    = size(Utf)[1]
    nf       = 2n_orb
    Uijkl_tf = zeros(size(Uijkl)...)
    Utf4x4 = kron(Utf, Utf)
    for i = 1:2
        for j = 1:2
            Uijkl_tf[:,:,:,:,i,j,j,i] = reshape(Utf4x4' * 
                                                reshape(Uijkl[:,:,:,:,i,j,j,i],nf,nf) * 
                                                Utf4x4, n_orb, n_orb, n_orb, n_orb)
        end
    end
    return Uijkl_tf
end
