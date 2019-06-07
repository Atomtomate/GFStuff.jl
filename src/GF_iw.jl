export DiscreteFunction, F, F_k, F_iω, F_τ, F_kiω


struct iω_grid{T} <: DiscreteFunction{T}
    β::Real
    f_grid::Array{T}
end

iω(β, n::Int) = 1.0im*(2.0 *n + 1)*π/β

iω(grid::iω_grid, n::Integer) = n < 0 ? -1.0*grid[-n] : grid[n+1]

function iω_array(β::Real, grid::Array{Int})
    fac = π/β
    res =  Array{Complex{Float64}}(undef, length(grid))
    for i = 1:length(grid)
        res[i] =  1.0im*((2.0 *grid[i] + 1)*fac)
    end
    return res
end

function iω_array(β::Real, size::Integer)
    grid = Array{Complex{Float64}}(undef, size)
    fac = π/β
    for i = 0:(size-1)
        grid[i+1] = 1.0im*((2.0 *i + 1)*fac)
    end
    return grid
end

function iω_grid(β::Real, size::Integer)
    return iω_grid(β, iω_array(β, size))
end

struct F_iω{T} <: DiscreteFunction{T}
    β::Real
    f_grid::Array{T}
end

struct F_τ{T} <: DiscreteFunction{T}
    β::Real
    τ_grid::Union{AbstractRange, Array{Float64}}
    f_grid::Array{T}
end

struct F_kτ{T} <: DiscreteFunction{T}
    k_grid::Union{AbstractRange, Array{Float64}}
    τ_grid::Union{AbstractRange, Array{Float64}}
    f_grid::Array{T}
end

struct F_kiω{T} <: DiscreteFunction{T}
    β::Real
    k_grid::Union{AbstractRange, Array{Float64}}
    f_grid::Array{T}
end

# ----- F_kiw -----
# ~~~~ array access ~~~~ 
Base.getindex(g::iω_grid, i::Int)   = g.f_grid[i];

# ~~~~~ math ~~~~~ 
Base.:+(lhs::F_kiω, rhs::Number)    = F_kiω(lhs.β, lhs.k_grid, lhs.f_grid .+ rhs);
Base.:+(lhs::Number, rhs::F_kiω)    = rhs + lhs;
Base.:+(lhs::F_kiω, rhs::F_kiω)     = F_kiω(lhs.β, lhs.k_grid, lhs.f_grid .+ rhs.f_grid);
Base.:-(lhs::F_kiω, rhs::Number)    = lhs + (-rhs);
Base.:-(lhs::Number, rhs::F_kiω)    = F_kiω(lhs.β, lhs.k_grid, lhs .- rhs.f_grid);
Base.:-(lhs::F_kiω, rhs::F_kiω)     = F_kiω(lhs.β, lhs.k_grid, lhs.f_grid .- rhs.f_grid);
Base.:*(lhs::F_kiω, rhs::Number)    = F_kiω(lhs.β, lhs.k_grid, lhs.f_grid .* rhs);
Base.:*(lhs::Number, rhs::F_kiω, )  = rhs * lhs;
Base.:*(lhs::F_kiω, rhs::F_kiω)     = F_kiω(lhs.β, lhs.k_grid, lhs.f_grid .* rhs.f_grid);

Base.:+(lhs::F_iω, rhs::Number)     = F_iω( lhs.β, lhs.f_grid .+ rhs);
Base.:+(lhs::Number, rhs::F_iω)     = rhs + lhs;
Base.:+(lhs::F_iω, rhs::F_iω)       = F_iω( lhs.β, lhs.f_grid .+ rhs.f_grid);
Base.:-(lhs::F_iω, rhs::Number)     = F_iω( lhs.β, lhs.f_grid .- rhs);
Base.:-(lhs::Number, rhs::F_iω)     = F_iω( lhs.β, lhs .- rhs.f_grid);
Base.:-(lhs::F_iω, rhs::F_iω)       = F_iω( lhs.β, lhs.f_grid .- rhs.f_grid);
Base.:*(lhs::F_iω, rhs::Number)     = F_iω( lhs.β, lhs.f_grid .* rhs);
Base.:*(lhs::Number, rhs::F_iω)     = rhs * lhs;
Base.:*(lhs::F_iω, rhs::F_iω)       = F_iω( lhs.β, lhs.f_grid .* rhs.f_grid);


Base.:+(lhs::iω_grid, rhs::Number)  = iω_grid(lhs.β, lhs.f_grid .+ rhs);
Base.:+(lhs::Number, rhs::iω_grid)  = rhs + lhs;
Base.:+(lhs::iω_grid, rhs::iω_grid) = iω_grid(lhs.β, lhs.f_grid .+ rhs.f_grid);
Base.:-(lhs::iω_grid, rhs::Number)  = lhs + (-rhs);
Base.:-(lhs::Number, rhs::iω_grid)  = iω_grid(rhs.β, lhs .- rhs.f_grid);
Base.:-(lhs::iω_grid, rhs::iω_grid) = iω_grid(lhs.β, lhs.f_grid .- rhs.f_grid);
Base.:*(lhs::iω_grid, rhs::Number)  = iω_grid(lhs.β, lhs.f_grid .* rhs);
Base.:*(lhs::Number, rhs::iω_grid)  = rhs * lhs;
Base.:*(lhs::iω_grid, rhs::iω_grid) = iω_grid(lhs.β, lhs.f_grid .* rhs.f_grid);
Base.:/(lhs::iω_grid, rhs::Number)  = iω_grid(lhs.β,  lhs.f_grid ./ rhs);
Base.:/(lhs::Number, rhs::iω_grid)  = iω_grid(rhs.β, lhs ./ rhs.f_grid);
Base.:/(lhs::iω_grid, rhs::iω_grid) = iω_grid(lhs.β, lhs.f_grid ./ rhs.f_grid);

inv_arr(F::F_iω, start, stop) = mapslices(LinearAlgebra.inv, F.f_grid[start:stop, :, :], dims=(2,3))
inv_arr(F::F_iω) = mapslices(LinearAlgebra.inv, F.f_grid, dims=(2,3))
inv(F::F_iω, start, stop)     = F_iω(F.β, mapslices(LinearAlgebra.inv, F.f_grid[start:stop, :, :], dims=(2,3)))
inv(F::F_iω)     = F_iω(F.β, mapslices(LinearAlgebra.inv, F.f_grid, dims=(2,3)))

rotate(F::F_iω, U::AbstractArray{<:Number, 2})  = F_iω(F.β, mapslices(M -> U' * M * U, F.f_grid, dims=(2,3)))
rotate(F::F_kiω, U::AbstractArray{<:Number, 2}) = F_kiω(F.β, F.k_grid, mapslices(M -> U' * M * U, F.f_grid, dims=(3,4)))

function iωProd(inp::F_kiω, fiw)
    grid = inp.f_grid
    for i ∈ 1:size(grid)[2]
        grid[:,i,:,:] = view(grid,:,i,:,:) .* fiw[i]
    end
    return F_kiω(inp.β, inp.k_grid, grid)
end

function iωProd(inp::F_iω, fiw)
    grid = inp.f_grid
    for i in 1:size(grid)[1]
        grid[i,:,:] = view(grid,i,:,:) .* fiw[i]
    end
    return F_iω(inp.β, inp.ω_grid, grid)
end

function iωProd(inp::F_k, fiw)
    grid = inp.f_grid
    length_k = size(inp.k_grid)[1]
    length_ω = size(fiw)[1]
    length_F = size(grid)[1]
    new_grid = Array{typeof(grid[1,1,1])}(undef, length_k, length_ω, length_F, length_F)
    for i in 1:size(grid)[2]
        new_grid[:,i,:,:] = grid .* fiw[i]
    end
    return F_kiω(inp.β, inp.k_grid, grid)
end
