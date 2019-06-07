module GFStuff
    __precompile__(false)
    include("../../../RootFinding/src/RootFinding.jl")
    using .RootFinding

    include("./GF.jl")
    include("./GF_iw.jl")
    include("./utility.jl")
    include("./GFStuff_iw.jl")

    import JLD
    using LinearAlgebra
    using Base.Filesystem
    using StaticArrays
    using Roots
    using ForwardDiff
    using Statistics

    using SharedArrays
    using Distributed

    function nf(ω::Array{Float64}, β::Float64, μ::Real)
        map(x -> nf(x, β, μ), ω)
    end

    function nf(ω::Float64, β::Float64, μ::Real)
        res = 0.0
        if β ≈ 0
            res = ω < μ ? 1.0 : 0.0
        else
            res = 1.0/(exp(β*(ω-μ)) + 1)
        end
        return res
    end

    function read_hk(fname, spinresolved = true)
        local nkpoints::Int64 = 0
        local nWann::Int64 = 0
        local nBands::Int64 = 0
        open(fname) do f
            lines   = readlines(f)
            tmp     = parse.(Int64,split(lines[1])[1:3])
            nkpoints, nWann, nBands = tmp
            kpoints = Array{Float64}(undef, nkpoints, 3)
            local hk = Array{Complex{Float64}}(undef, nkpoints, nWann, nWann)
            for i = 1:nkpoints
                kpoints[i, :] = parse.(Float64, split(lines[(i-1)*(nWann+1) + 2]))
                for j = 1:nWann
                    hk[i, j, :] = conv_FortranStrToComplex(lines[(i-1)*(nWann+1) + 2+j])
                end
            end 
            local success::Bool = true
            res = F_k(kpoints, hk)
            #JLD.save("hk.jld", "Hk", res)
            return res
        end
    end

    # ---- Real Freq GF
    function ALoc_from_H(Hₖ::F_k, δ::Real, μ::Real, ω::T) where T<:StepRangeLen
        H_grid = Hₖ.f_grid
        size_F = size(H_grid)
        A = SharedArray{Float64}((length(ω), size_F[end-1], size_F[end]))
        tmp = (ω .+ δ*im .+ μ)
        mw = length(ω)
        #println("0 %")
        @sync @distributed for i in 1:mw
            A[i,:,:] .= 0.0
            for j in 1:size_F[1]
                tmp2 = -imag(LinearAlgebra.inv(tmp[i]*Matrix(I, size_F[end-1], size_F[end]) - view(H_grid, j,:,:)))
                A[i,:,:] += tmp2 ./ (π * size_F[1])
            end
            #if i%div(mw,10) == 0
            #    print("\r")
            #    print(100*i/mw)
            #    println("%")
            #end
        end
        return F_ω(ω, sdata(A))
    end

    function ALoc_from_H_dbg(Hₖ::F_k, δ::Real, μ::Real, ω::T, Σ) where T<:StepRangeLen
        H_grid = Hₖ.f_grid
        size_F = size(H_grid)
        A = zeros(Float64, length(ω), size_F[end-1], size_F[end])
        tmp = (ω .+ δ*im .+ μ)
        mw = length(ω)
        for i in 1:mw
            for j in 1:size_F[1]
                tmp2 = imag(LinearAlgebra.inv(tmp[i]*Matrix(I, size_F[2], size_F[3]) - Σ - view(H_grid, j,:,:)))
                A[i,:,:] += tmp2 ./ (π * size_F[1])
            end
            if (mw/10) % i == 0
                println(mw/i)
            end
        end
        return F_ω(ω, A)
    end

    function G_from_H(Hₖ::F_k, δ::Real, μ::Real, ω::T) where T<:StepRangeLen
        #G_from_H(Hₖ, δ, μ, ω, Σ)
        H_grid = Hₖ.f_grid
        size_Hₖ = size(H_grid)
        Σ = zeros(Complex{Float64}, size_Hₖ[end-1], size_Hₖ[end])
        return G_from_H(Hₖ, δ, μ, ω, Σ)
    end

    function G_from_H(F::Array{<:Number}, δ::Real, μ::Real, ω::T) where T<:StepRangeLen
        size_F = size(F)
        length_ω = length(ω)
        tmp = (ω .+ δ*im .+ μ)
        G = Array{Complex{Float64}}(undef, length_ω, size_F[end-1], size_F[end])
        for i in 1:length(ω)
            G[i,:,:] = LinearAlgebra.inv(tmp[i]*Matrix(I, size_F[1], size_F[2]) - F)
        end
        return F_ω(ω, G)
    end

    function G_from_H(Hₖ::F_k, δ::Real, μ::Real, ω::T, Σ) where T<:StepRangeLen
        H_grid = Hₖ.f_grid
        size_Hₖ = size(H_grid)
        length_ω = length(ω)
        length_k = size(Hₖ.k_grid)[1]
        tmp = (ω .+ δ*im .+ μ)
        G = Array{Complex{Float64}}(undef, length_k, length_ω, size_Hₖ[end-1], size_Hₖ[end])
        for i in 1:length(ω)
            tmp2 = tmp[i]*Matrix(I, size_Hₖ[end-1], size_Hₖ[end]) - Σ
            for j in 1:size_Hₖ[1]
                G[j,i,:,:] = LinearAlgebra.inv(tmp2 - view(H_grid, j, :, :))
            end
        end
        return F_kω(Hₖ.k_grid, ω, G)
    end

    function A_from_G(G::F_kω)
        res::F_kω = F_kω(G.k_grid, G.ω_grid, -imag.(G.f_grid) ./ π)
        return res
    end

    function A_from_G(G::F_k)
        F_kω(G.k_grid, -imag.(G.f_grid) ./ π)
    end
    
    function filling(G::F_kω, μ::Real = 0.0)
        A = A_from_G(G)
        #TODO: FIX FACTOR 2: read_hk should duplicate Hk
        res = 2.0*kSum(ωInt(ωProd(A, nf(collect(A.ω_grid), 0.0, μ)))).f_grid
        return res
    end

    function filling(ALoc::F_ω, μ::Real = 0.0)
        #TODO: FIX FACTOR 2: read_hk should duplicate Hk
        res = 2.0*ωInt(ωProd(ALoc, nf(collect(ALoc.ω_grid), 0.0, μ))).f_grid
        return res
    end
    
    function check_norm(A::F_kω)
        ωInt(kSum(sum(A, dims=[3,4]))).f_grid
    end

    function adjust_μ(Hₖ, δ::Real, ω, target::Real, μ_initial::Array{Float64,1} = [0.0, 2.0], tol::Real = 0.001)
        #μ::Real = μ_initial 
        #G = G_from_H(Hₖ, δ, μ, ω)
        function adjust(μ::Real)
            println("Starting matrix inversion")
            #G = G_from_H(Hₖ, δ, μ, ω)
            #na = filling(G)
            Aloc = ALoc_from_H(Hₖ, δ, μ, ω .- μ)
            na = filling(Aloc, μ)
            n::Real =  LinearAlgebra.tr(na)
            print("\nMatrix inversion done. mu = ")
            print(μ)
            print(", n = ")
            println(n)
            return abs(n)-target
        end
        #μ = RootFinding.find_zero(adjust, Roots.fzero(adjust, -13.14))# order=1)
        μ = RootFinding.brent(adjust, μ_initial... ,tol, tol, 30)# order=1)
        return μ
    end

    function compute_Eₖ(Hₖ, verbose=false)
        length_k = size(Hₖ.f_grid)[1]
        n_bands = size(Hₖ.f_grid)[2]
        Eₖ = Array{Float64}(undef, length_k, n_bands) 
        for i in 1:length_k
            if verbose && (i%div(length_k,3) == 0)
                println(100*i/length_k)
            end
            Eₖ[i,:] = LinearAlgebra.eigvals(LinearAlgebra.Hermitian(Hₖ.f_grid[i,:,:]))
        end
        return Eₖ
    end

end
