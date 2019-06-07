using LinearAlgebra
using Roots
using SharedArrays
using Distributed
using Optim
using Printf


# ---- Im Freq GF
function G0_from_H(Hₖ::F_k, β::Real, μ::Real, nFreq::Int)
    Σ = zeros(Float64, nFreq, size(Hₖ.f_grid)[end-1], size(Hₖ.f_grid)[end])
    GLoc_from_H(Hₖ, Σ, β, μ)
end

GLoc_from_H(Hₖ::F_k, Σ::F_iω, β::Real, μ::Real) = GLoc_from_H(Hₖ, Σ.f_grid, β, μ)

function GLoc_from_H(Hₖ::F_k, Σ::Array, β::Real, μ::Real) where T<:StepRangeLen
    nFreq = size(Σ)[1]
    H_grid = Hₖ.f_grid
    Nₖ = size(Hₖ.f_grid)[1]
    N = size(Hₖ.f_grid)[end]
    G = SharedArray{Complex{Float64}}(nFreq, N, N)
    iwn = iω_grid(β, nFreq)
    tmp = (iwn.f_grid .+ μ)
    @sync @distributed for i in 1:nFreq
        G[i,:,:] .= 0.0
        for j in 1:Nₖ
            tmp2 = LinearAlgebra.inv(tmp[i]*Matrix(I, N, N) .- view(H_grid, j,:,:) .- view(Σ,i,:,:))
            G[i,:,:] += tmp2
        end
        G[i,:,:] = G[i,:,:] ./ Nₖ
    end
    return F_iω(β, sdata(G))
end

function print_HLoc(Hₖ::F_k)
    H_grid = Hₖ.f_grid
    Nₖ = size(Hₖ.f_grid)[1]
    N = size(Hₖ.f_grid)[end]
    HLoc = SharedArray{Complex{Float64}}(N, N)
    HLoc[:,:] .= 0.0
    for j in 1:Nₖ
        HLoc[:,:] += Hₖ.f_grid[j,:,:] ./ Nₖ
    end
    println("Local Hamiltonian: ")
    println(HLoc)
end


function filling(Hₖ::F_k, β::Real, μ::Real, nFreq::Int)
    Σ = GFStuff.F_iω(β, zeros(Complex{Float64}, nFreq, size(Hₖ.f_grid)[end-1], size(Hₖ.f_grid)[end]))
    return filling(Hₖ, Σ, β, μ, nFreq)
end

function filling(Hₖ::F_k, Σ::F_iω, β::Float64, μ::Float64)
    filling(Hₖ, Σ.f_grid, β, μ)
end

function filling(Hₖ::F_k, Σ::Array, β::Float64, μ::Float64)
    nFreq = size(Σ)[1]
    iwn = iω_grid(β, nFreq)
    Gl = GLoc_from_H(Hₖ, Σ, β, μ)
    N = size(Gl.f_grid)[end]
    res = 0.0                                                   # Tail fit
    for i in 1:nFreq
        res += real( tr(view(Gl.f_grid,i, :, :) .- 1.0/iwn[i])) 
    end
    # 0.5*N: Tail correction after trace
    res = 2*res/β + 0.5*N
    return res
end

function adjust_μ(Hₖ, Σ::F_iω, β::Real, target::Real, μ_initial::Array{Float64,1} = [0.0, 2.0], tol::Real = 1e-12)
    adjust_μ(Hₖ, Σ.f_grid, target, μ_initial, tol)
end

function adjust_μ(Hₖ, Σ::Array, β::Real, target::Real, μ_initial::Array{Float64,1} = [0., 2.0], tol::Real = 1e-12)
    nFreq = size(Σ)[1]
    println("Starting matrix inversion")
    function adjust(μ::Real)
        n = filling(Hₖ, Σ, β, μ)
        print("\rMatrix inversion done. mu = $μ, N_tot = $n")
        return n-target
    end
    println("\r")
    #TODO: find range automatically
    μ = RootFinding.brent(adjust, μ_initial... ,tol, tol, 40)# order=1)
    println("\rFound μ = $μ")
    return μ
end

#TODO: range for tau
function ω_to_τ(F::F_iω, tail_coeff::Union{Array{Real}, Nothing} = nothing, n_tail = 5, nFreq = 20, iω_cut = :end)
    N = size(F.f_grid)[2]
    Nᵢ = iω_cut == :end ? size(F.f_grid)[1] : iω_cut
    N_τ = 4*Nᵢ + 1
    tail_coeff = tail_coeff == nothing ? tail(F, n_tail, nFreq, iω_cut) : tail_coeff

    τ_list  = collect(F. β.*(0:(N_τ-1))./(N_τ-1))
    τ_list[1] += N_τ/100.0 
    τ_list[end] -= N_τ/100.0
    iω_list = iω_array(F.β, Nᵢ)
    ft_grid = exp.(τ_list * iω_list')

    res = Array{Complex{Float64}}(undef, N_τ, N, N)
    for i = 1:N
        for j = i:N
            tail_ω_ij = tail_func(collect(0:(Nᵢ-1)), F.β, tail_coeff[i,j,:] )
            tail_ω_ji = tail_func(collect(0:(Nᵢ-1)), F.β, tail_coeff[j,i,:] )
            tail_τ = tail_τ_func(τ_list, F.β, tail_coeff[i,j,:] )
            data_ij = view(F.f_grid, :, i, j) .- tail_ω_ij 
            data_ji = view(F.f_grid, :, j, i) .- tail_ω_ji
            for ti = 1:N_τ
                res[ti, i, j] = sum((data_ij .* view(ft_grid, ti, :)))               
                res[ti, i, j] += sum(conj.( (data_ji .* view(ft_grid, ti, :)) ))
            end
            res[:,i,j] = view(res,:,i,j) ./ F.β .+ tail_τ
            if i != j
                res[:,j,i] = conj(view(res, :, i, j))
            end
        end
    end
    return F_τ(F.β, τ_list, res)
end


function τ_to_ω(F::F_τ, tail::Union{Array{Real}, Nothing} = nothing)
    N_τ = size(F.f_grid)[2]
    N_ω = Nᵢ
    tail_coeff = tail_coeff == nothing ? tail(F, 5) : tail_coeff

    τ_list  = F.τ_list
    iω_list = iω_array(F.β, Nᵢ)
    ft_grid = exp.(τ_list * iω_list')
    τ_list[1] += 1e-8; τ_list[end] -= 1e-8
    #TODO: not needed for now, implement later

    return F
end


GBath_from_GLoc(GLoc::F_iω, Σ::F_iω)   = F_iω(GLoc.β, GBath_from_GLoc(GLoc.f_grid, Σ.f_grid))
GBath_from_GLoc(GLoc::F_iω, Σ::Array)  = F_iω(GLoc.β, GBath_from_GLoc(GLoc.f_grid, Σ))
GBath_from_GLoc(GLoc::Array, Σ::Array) = mapslices( LinearAlgebra.inv, mapslices( LinearAlgebra.inv, GLoc, dims=(2,3)) .+ Σ, dims=(2,3) )

function Δ_from_G(G::F_iω, μ::Array, Σ::F_iω)
    return Δ_from_G(G, μ, Σ.f_grid)
end

function Δ_from_G(G::F_iω, μ::Array, Σ::Array)
    N_ω = size(G.f_grid)[1]
    N =  size(G.f_grid)[end]
    Δ = Array{Complex{Float64}}(undef, N_ω, N, N)
    iwn = iω_grid(G.β, N_ω)
    if N == 1
        for i = 1:N_ω
            Δ[i,:,:] = iwn[i]*Matrix(I, N, N) .+ μ  .- 1.0/(G.f_grid[i,:,:]) .- view(Σ, i, :, :)
        end
    else
        for i = 1:N_ω
            # Δ(iωₙ) = iωₙ + μ - G⁻¹_loc(iωₙ) - Σ(iωₙ)
            Δ[i,:,:] = iwn[i]*Matrix(I, N, N) .+ μ  .- LinearAlgebra.inv(G.f_grid[i,:,:]) .- view(Σ, i, :, :) 
        end
    end
    return F_iω(G.β, Δ)
end

function G_from_H(Hₖ::F_k, β::Real, μ::Real, nFreq::Integer)
    Σ = GFStuff.F_iω(β, zeros(Complex{Float64}, nFreq, size(Hₖ.f_grid)[end-1], size(Hₖ.f_grid)[end]))
    return G_from_H(Hₖ, β, μ, nFreq, Σ)
end

function G_from_H(Hₖ::F_k, β::Real, μ::Real, nFreq::Integer, Σ::F_iω)
    Nₖ = size(Hₖ.k_grid)[1]
    N = size(Hₖ.f_grid)[end]
    iwn = iω_grid(β, nFreq)
    tmp = (iwn.f_grid .+ μ)
    G = Array{Complex{Float64}}(undef, Nₖ, nFreq, N, N)
    for i in 1:nFreq
        tmp2 = tmp[i]*Matrix(I, N, N) .- view(Σ.f_grid, i, :, :) 
        for j in 1:Nₖ
            G[j,i,:,:] = LinearAlgebra.inv(tmp2 - view(Hₖ.f_grid, j, :, :))
        end
    end
    return F_kiω(β, Hₖ.k_grid, G)
end


function tail_func(iωn::Array{Complex{Float64}}, β, c::Array{Float64})
    res = [c[1] for i = 1:length(iωn)]
    for  i = 2:length(c)
        res = res .+ c[i]./(iωn .^ (i-1))
    end
    return res
end

function tail_func(n::Array{Int}, β, c::Array{Float64})
    iωn = iω_array(β, n)
    tail_func(iωn, β, c)
end

function tail_τ_func(τ::Array, β, c::Array{Float64})
    res = [c[1] for i = 1:length(τ)]
    for  i = 2:length(c)
        if i == 2
            res = res .- (c[2]/2)
        elseif i == 3 
            res = res .+ (c[3]/4) .* (2 .* τ .- β)
        elseif i == 4 
            res = res .+ (c[4]/4) .* (τ .* (β .- τ))
        elseif i == 5 
            res = res .+ (c[5]/48) .* (2 .* τ .- β) .* (2 .* τ .* τ .- 2 .* β .* τ .- (β*β))
        else  
            @printf("Warning: only 4 tail coefficients implemented, supplied: %d\n", length(c))
        end

    end
    return res
end

tail_real_func(n::Array{Int}, β, c::Array{Float64}) = [c[1] for i = 1:length(n)] .+ c[2] ./ (iω_array(β, n) .^ 2) .+ c[3] ./ (iω_array(β, n) .^ 4)

#TODO: -25  statt 480
function tail(G::F_iω, n_tail = 5, nFreq = 20, stop = :end)
    stop = stop == :end ? size(G.f_grid)[1] : stop
    N = size(G.f_grid)[2]
    start = stop - nFreq + 1
    β = G.β
    iωn = iω_array(β, collect(start:stop))
    res = zeros(Float64, N, N, n_tail)
    for i = 1:N
        for j = i:N
            g_grid = view(G.f_grid, start:stop, i, j)
            cost(c) = sum(abs2.(imag(g_grid) - imag(tail_func(iωn, β, c))))
            res[i,j,:] = Optim.minimizer(Optim.optimize(cost, zeros(n_tail), Optim.BFGS()))
            res[j,i,:] = view(res,i,j,:)
            #res[i,j,:] = [0.0 1.0 0.0 0.5 0.0]
            #res[j,i,:] = [0.0 1.0 0.0 0.5 0.0]
        end
    end
    return res
end

function μ_loc(G::F_iω, nFreq = 20, stop = :end)
    stop = stop == :end ? size(G.f_grid)[1] : stop
    N = size(G.f_grid)[2]
    β = G.β
    start = stop - nFreq + 1
    iωn = iω_array(β, collect(start:stop))
    res = Array{Float64}(undef, N, N, 5)
    g_inv = inv_arr(G, start, stop)
    for i = 1:N
        for j = i:N
            g_grid = view(g_inv, :, i, j)
            cost(c) = sum(abs2.(g_grid - tail_func(iωn, β, c)))
            res[i,j,:] = Optim.minimizer(Optim.optimize(cost, zeros(5), Optim.BFGS()))
            res[j,i,:] = view(res,i,j,:)
        end
    end
    println("in mu loc")
    println(res)
    println(" last freq: ")
    println(g_inv[end,:,:])
    return res[:,:,1]
end
