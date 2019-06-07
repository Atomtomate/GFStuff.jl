using StaticArrays

export DiscreteFunction, F, F_k, F_ω, F_kω

abstract type DiscreteFunction{T <: Number} end

struct F{T} <: DiscreteFunction{T}
    f_grid::Array{T}
end

struct F_k{T} <: DiscreteFunction{T}
    k_grid::Union{AbstractRange, Array{<:Real}}
    f_grid::Array{T}
end

struct F_ω{T} <: DiscreteFunction{T}
    ω_grid::Union{AbstractRange{<:Real}, Array{<:Real}}
    f_grid::Array{T}
end

struct F_kω{T} <: DiscreteFunction{T}
    k_grid::Union{AbstractRange, Array{<:Real}}
    ω_grid::Union{AbstractRange{<:Real}, Array{<:Real}}
    f_grid::Array{T}
end

# ===== Functions =====
Base.:+(lhs::F_kω, rhs::Number) = F_kω(lhs.k_grid, lhs.ω_grid, lhs.f_grid .+ rhs)
Base.:+(lhs::F_kω, rhs::F_kω)   = F_kω(lhs.k_grid, lhs.ω_grid,  lhs.f_grid .+ rhs.f_grid)
Base.:+(lhs::Number, rhs::F_kω) = rhs + lhs
Base.:-(lhs::F_kω, rhs::Number) = lhs + (-lhs)
Base.:-(lhs::F_kω, rhs::F_kω)   = F_kω(lhs.k_grid, lhs.ω_grid, lhs.f_grid .- rhs.f_grid)
Base.:-(lhs::Number, rhs::F_kω) = F_kω(lhs.k_grid, lhs.ω_grid, lhs .- lhs.f_grid )
Base.:*(lhs::F_kω, rhs::Number) = F_kω(lhs.k_grid, lhs.ω_grid, lhs.f_grid .* rhs)
Base.:*(lhs::Number, rhs::F_kω) = rhs * lhs
Base.:*(lhs::F_kω, rhs::F_kω)   = F_kω(lhs.k_grid, lhs.ω_grid,  lhs.f_grid .* rhs.f_grid)

# ==== Operations on external Grids ====
# ---- F_kw ----
function ωProd(inp::F_kω, fw)
    grid = inp.f_grid
    for i ∈  1:size(grid)[2]
        grid[:,i,:,:] = view(grid,:,i,:,:) .* fw[i]
    end
    return F_kω(inp.k_grid, inp.ω_grid, grid)
end

function ωProd(inp::F_ω, fw)
    grid = inp.f_grid
    for i in 1:size(grid)[1]
        grid[i,:,:] = view(grid,i,:,:) .* fw[i]
    end
    return F_ω(inp.ω_grid, grid)
end

function ωProd(inp::F_k, fw)
    grid = inp.f_grid
    length_k = size(inp.k_grid)[1]
    length_ω = size(fw)[1]
    length_F = size(grid)[1]
    new_grid = Array{typeof(grid[1,1,1])}(undef, length_k, length_ω, length_F, length_F)
    for i in 1:size(grid)[2]
        new_grid[:,i,:,:] = grid .* fw[i]
    end
    return F_kω(inp.k_grid, inp.ω_grid, grid)
end

kSum(inp::Array) = collapseDim(inp, 1)./size(inp)[1]
kSum(inp::F_kω)  = F_ω(inp.ω_grid, collapseDim(inp.f_grid, 1)./size(inp.k_grid)[1])
kSum(inp::F_k)   = F(collapseDim(inp.f_grid, 1)./size(inp.k_grid)[1])
kSum(inp::F_ω)   = inp
kSum(inp::F)     = inp

ωInt(inp::F_kω)  = F_k(inp.k_grid, riemannSum(inp.f_grid, inp.ω_grid, 2))
ωInt(inp::F_ω)   = F(riemannSum(inp.f_grid, inp.ω_grid, 1))
ωInt(inp::F_k)   = inp
ωInt(inp::F)     = inp

function bandSum_internal(inp) 
    last = length(size(inp))
    res = dropdims(dropdims(sum(inp, dims=[last-1, last]), dims=last), dims=last-1)
    return res
end
bandSum(inp::F_kω) = F_kω(inp.k_grid, inp.ω_grid, bandSum_internal(inp.f_grid))
bandSum(inp::F_k)  = F_k(inp.k_grid, bandSum_internal(inp.f_grid))
bandSum(inp::F_ω)  = F_ω(inp.ω_grid, bandSum_internal(inp.f_grid))
bandSum(inp::F)    = F(bandSum_internal(inp.f_grid))

rotate(F::F_k, U::AbstractArray{<:Number, 2}) = F_k(F.k_grid, mapslices(M -> U' * M * U, F.f_grid, dims=(2,3)))
rotate(F::AbstractArray{<:Number, 3}, U::AbstractArray{<:Number, 2}) = mapslices(M -> U' * M * U, F, dims=(2,3))
rotate(F::AbstractArray{<:Number, 2}, U::AbstractArray{<:Number, 2}) = U' * F * U
