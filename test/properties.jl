using Test
using Catalyst: @reaction_network, reactions
using FiniteStateProjection
using OrdinaryDiffEq: Tsit5
using CommonSolve: solve
using Distributions: Poisson, pdf
using LinearAlgebra
using SparseArrays
using SciMLBase: ODEProblem, SteadyStateProblem
using SteadyStateDiffEq: SSRootfind

@testset "Structural and physical properties" begin
    @testset "one rate function per reaction" begin
        rs = @reaction_network begin
            k1, 0 --> A
            k2, A --> 0
            k3, A + B --> 2B
        end
        sys = FSPSystem(rs, [:A, :B])
        @test length(sys.rfs) == length(reactions(rs))
    end

    @testset "steady-state matrix conserves probability" begin
        rs = @reaction_network begin
            k1, 0 --> A
            k2, A --> 0
            k3, A --> A + B
            k4, B --> 0
        end
        sys = FSPSystem(rs, [:A, :B])
        pmap = [:k1 => 1.0, :k2 => 0.5, :k3 => 0.3, :k4 => 0.7]
        A_ss = SparseMatrixCSC(sys, (12, 12), pmap, SteadyState())
        @test all(abs.(sum(A_ss; dims = 1)) .< 1.0e-12)
    end

    @testset "time-dependent matrix never creates probability" begin
        rs = @reaction_network begin
            k1, 0 --> A
            k2, A --> 0
        end
        sys = FSPSystem(rs, [:A])
        pmap = [:k1 => 3.0, :k2 => 1.0]
        A = SparseMatrixCSC(sys, (15,), pmap, 0.0)
        # Probability can only leave the truncated state space, so column sums are <= 0.
        @test all(sum(A; dims = 1) .<= 1.0e-12)
    end

    @testset "ODE dynamics conserve probability away from the boundary" begin
        rs = @reaction_network begin
            k1, 0 --> A
            k2, A --> 0
        end
        sys = FSPSystem(rs, [:A])
        pmap = [:k1 => 1.0, :k2 => 1.0]
        u = zeros(60)
        u[1] = 1.0
        du = similar(u)
        prob = ODEProblem(sys, u, (0.0, 1.0), pmap)
        prob.f(du, u, prob.p, 0.0)
        @test sum(du) ≈ 0 atol = 1.0e-12
    end

    @testset "ODE solution matches the matrix exponential" begin
        rs = @reaction_network begin
            k1, 0 --> A
            k2, A --> 0
            k3, A --> A + B
            k4, B --> 0
        end
        sys = FSPSystem(rs, [:A, :B])
        pmap = [:k1 => 1.0, :k2 => 0.5, :k3 => 0.3, :k4 => 0.7]
        dims = (10, 10)
        u0 = zeros(dims)
        u0[1, 1] = 1.0
        t = 1.5

        A = Matrix(SparseMatrixCSC(sys, dims, pmap, 0.0))
        expected = reshape(exp(A * t) * vec(u0), dims)

        prob = ODEProblem(sys, u0, (0.0, t), pmap)
        sol = solve(prob, Tsit5(), abstol = 1.0e-10, reltol = 1.0e-10)
        @test sol.u[end] ≈ expected atol = 1.0e-7
    end

    @testset "steady state of a birth-death process is Poisson" begin
        rs = @reaction_network begin
            k, 0 --> A
            d, A --> 0
        end
        sys = FSPSystem(rs, [:A])
        pmap = [:k => 3.0, :d => 1.0]
        N = 40
        u0 = zeros(N + 1)
        u0[1] = 1.0

        prob = SteadyStateProblem(sys, u0, pmap)
        sol = solve(prob, SSRootfind())
        p = sol.u ./ sum(sol.u)
        @test p ≈ pdf.(Poisson(3.0), 0:N) atol = 1.0e-6
    end

    @testset "parameters are looked up by name, not by position" begin
        rs = @reaction_network begin
            k1, 0 --> A
            k2, A --> 0
        end
        sys = FSPSystem(rs, [:A])
        forward = [:k1 => 2.0, :k2 => 1.0]
        reversed = [:k2 => 1.0, :k1 => 2.0]
        @test SparseMatrixCSC(sys, (8,), forward, 0.0) ≈
            SparseMatrixCSC(sys, (8,), reversed, 0.0)
    end
end
