include("../src/GFStuff.jl")

import JLD
using Base.Filesystem
using DelimitedFiles, CSV, DataFrames

@static if VERSION < v"0.07-DEV.2005"
    using Base.Test
else
    using Test
end

function gridDim(F_kω)
    collect(size(F_kω.f_grid))
end

@testset "GFStuff" begin
    println("Testing GF Stuff")
    Hₖ = GFStuff.read_hk("hk.w2wout")

    @testset "Utility" begin

        k_path = [0.0 0.0 0.0;
                  0.5 0.0 0.0;
                  0.5 0.5 0.0;
                  0.0 0.0 0.0;
                  0.5 0.5 0.5]
        @test_throws ArgumentError GFStuff.conv_FortranStrToComplex("")
        @test_throws UndefVarError GFstuff.conv_FortranStrToComplex("1.1")
        @test all(GFStuff.conv_FortranStrToComplex("1.1 0.1 103 3.1") .== [1.1+0.1im,103+3.1im])

        x1 = range(1.0,stop=3.0,length=10)
        x2 = range(1.0,stop=3.0,length=100)
        res1 = 3.0*3.0/2.0 - 1.0*1.0/2.0
        @test GFStuff.riemannSum(x1, x1, 1) ≈ res1
        @test GFStuff.riemannSum(x2, x2, 1) ≈ res1
        @test GFStuff.min_diff([0.0, 0.1, 0.1, 0.9, 0.91]) ≈ 0.01
        @test all(GFStuff.k_spacings(Hₖ) .≈ [0.125 0.125 0.125])
        @test all(GFStuff.Nk(Hₖ) .== [8 8 8])
        @test all(size(GFStuff.gen_kPath(Hₖ, k_path).k_grid) .== (17, 3))
        @test all(size(GFStuff.gen_kPath(Hₖ, k_path).f_grid) .== (17, 3, 3))
        #@test all(GFStuff.gen_kPath(Hₖ, k_path).k_grid .≈ [0.0 0.0 0.0; 0.125 0.0 0.0; 0.25 0.0 0.0; 0.375 0.0 0.0; 0.5 0.0 0.0; 0.5 0.125 0.0; 0.5 0.25 0.0; 0.5 0.375 0.0; 0.5 0.5 0.0; 0.375 0.375 0.0; 0.25 0.25 0.0; 0.125 0.125 0.0; 0.0 0.0 0.0])
    end

    @testset "IO" begin
        println(" - Testing IO")
        Hₖ_read = GFStuff.read_hk("hk.w2wout")
        Hₖ_exp = JLD.load(joinpath(@__DIR__ , "expectedResults/hk.jld"), "Hk")
        @test GFStuff.compare3d(Hₖ_read, Hₖ_exp)
    end

    @testset "GF" begin
        println(" - Testing GF")
        size_ω = 500
        range_ω = 10
        δ = 0.03
        μ = 0.0
        β = 0.0
        ω = range(-range_ω,stop=range_ω,length=size_ω)
        G = GFStuff.G_from_H(Hₖ, δ , μ , ω)
        size_k = size(Hₖ.k_grid)[1]

        @testset "nf" begin
            @test GFStuff.nf(-1.0, 0.0) == 1.0
            @test GFStuff.nf(0.0, 0.1) == 0.5
            @test GFStuff.nf(-1.0, 0.1) > 0.0
            @test GFStuff.nf(1.0, 0.1) > 0.0
        end

        @testset "GF" begin
            println("  . Testing GF container")
            f1 = GFStuff.F([1 2])
            f2 = GFStuff.F([1 2; 3 4.1])
            f3 = GFStuff.F([1+2im 2; 3 4])
            f4 = GFStuff.F_k([1. 2.],[1 2])
            f5 = GFStuff.F_k([1. 2.; 3. 4.],[1 2])
            f6 = GFStuff.F_k(1:1:5,[1+2im 2; 3 4])
            f7 = GFStuff.F_kω(1:1:5, [1,4,5],[1+2im 2; 3 4])
            f8 = GFStuff.F_iω(1.0, false, [1.0 13.0+2im 13.0-2im -3.0])
            f9 = GFStuff.F_kiω(1.0, false, 1:2, [1.0 13.0+2im; 13.0-2im -3.0])
            @test all(f1.f_grid .== [1 2])
            @test all(f2.f_grid .== [1 2; 3 4.1])
            @test all(f3.f_grid .== [1+2im 2; 3 4])
            @test all(f4.f_grid .== [1 2])
            @test all(f5.k_grid .== [1. 2.; 3. 4.])
            @test all(f6.f_grid .== [1+2im 2; 3 4])
            @test all(f7.f_grid .== [1+2im 2; 3 4])
            @test all((f7 * 7.0im).f_grid .== [-14.0+7.0im 0.0+14.0im; 0.0+21.0im 0.0+28.0im])
            @test all((f7 * f7).f_grid .== [-3+4im 4; 9 16])
            #@test all((f7 + 7.0im).f_grid .== [-14.0+7.0im 14.0im; 21im 28im])
            @test all((f7 + f7).f_grid .== [2+4im 4+0im; 6+0im 8+0im])
            @test all((f8 * 7.0im).f_grid .== [0.0+7.0im -14.0+91.0im 14.0+91.0im -0.0-21.0im])
            @test all((f8 * f8).f_grid .== [1.0+0.0im 165.0+52.0im 165.0-52.0im 9.0-0.0im])
            #@test all((f7 + 7.0im).f_grid .== [-14.0+7.0im 14.0im; 21im 28im])
            @test all((f9 + f9).f_grid .== [2.0+0.0im 26.0+4.0im; 26.0-4.0im -6.0+0.0im])
        end

        @testset "H to G" begin
            @test all(gridDim(G) .== [512, size_ω, 3, 3])
        end

        @testset "H to G" begin
            @test all(gridDim(G) .== [512, size_ω, 3, 3])
        end

        A = GFStuff.A_from_G(G)
        @testset "G to A" begin
            @test typeof(A) == Main.GFStuff.F_kω{Float64}
            @test typeof(A.f_grid) == Array{Float64,4}
            @test all(gridDim(G) .== gridDim(A) )
        end
        #println("test norm")
        #println(GFStuff.check_norm(A))
        #println(sum(GFStuff.check_norm(A), dims=[1,2]))

        G_loc = GFStuff.kSum(G)
        A_loc = GFStuff.kSum(A)

        @testset "k-sum" begin
            @test all(gridDim(G_loc) .== [size_ω, 3, 3])
            @test all(gridDim(A_loc) .== [size_ω, 3, 3])
        end

        writedlm("Aloc.csv", A_loc.f_grid)

        G2 = GFStuff.ωInt(G)
        G_loc2 = GFStuff.ωInt(G_loc)
        @testset "ω-Int" begin
            @test all(gridDim(G2) .== [size_k, 3, 3])
            @test all(gridDim(G_loc2) .== [3, 3])
        end

        @testset "bandSum" begin
            @test all(gridDim(GFStuff.bandSum(G)) .== [size_k, size_ω])
            @test all(gridDim(GFStuff.bandSum(G_loc)) .== [size_ω])
            @test typeof(GFStuff.bandSum(G_loc2).f_grid[1]) == Complex{Float64}
        end

        @testset "filling" begin
            println(" - Testing μ adjustment")
            #n = GFStuff.filling(G)
            #@test typeof(n) == Array{Float64, 2}
            ##n2 = GFStuff.filling(G_loc, ω, δ )
            #@test all(collect(size(n)) .== [3, 3])
        end
        μ_new = 0.0
        #μ_new = GFStuff.adjust_μ(Hₖ, δ , ω, 1.0, μ)
        @testset "adjust filling" begin
            #μ_new = GFStuff.adjust_μ(Hₖ, δ , ω, 1.0)
            #@test μ_new ≈ 7.936348
        end


        A_11 = A.f_grid[:,:,1,1]'
        writedlm("tmp.csv", A_11)
        A_22 = A.f_grid[:,:,2,2]'
        writedlm("tmp2.csv", A_22)
        A_33 = A.f_grid[:,:,3,3]'
        writedlm("tmp3.csv", A_33)
        G2 = GFStuff.G_from_H(Hₖ, δ , μ_new , ω)
        A2 = GFStuff.A_from_G(G2)
        ##println("test norm")
        ##println(GFStuff.check_norm(A2))
        ##println(sum(GFStuff.check_norm(A2), dims=[1,2]))

        A2_11 = A2.f_grid[:,:,1,1]'
        writedlm("tmp.2.csv", A2_11)
        A2_22 = A2.f_grid[:,:,2,2]'
        writedlm("tmp2.2.csv", A2_22)
        A2_33 = A2.f_grid[:,:,3,3]'
        writedlm("tmp3.2.csv", A2_33)

        @testset "HF test" begin

        end
    end
end
