using Test
using OffsetArrays: OffsetArray
using Catalyst: @reaction_network
using FiniteStateProjection
using SciMLBase: ODEProblem
using SparseArrays: SparseMatrixCSC

@testset "Offset of the molecular count zero" begin
    rs = @reaction_network begin
        birth, 0 --> A
        death, A --> 0
    end
    pmap = [:birth => 3.0, :death => 4.0]

    @testset "the offset is the array index of count zero" begin
        # With offset 2, index 3 is count 1, so the death propensity is death * 1.
        sys = FSPSystem(rs, OrderedIndexHandler(rs; offset = 2))
        @test sys.rfs[2]([3], 0.0, 3.0, 4.0) == 4.0
        @test sys.rfs[2]([2], 0.0, 3.0, 4.0) == 0.0
    end

    @testset "the default offset is 1" begin
        sys = FSPSystem(rs)
        @test sys.rfs[2]([2], 0.0, 3.0, 4.0) == 4.0
        @test sys.rfs[2]([1], 0.0, 3.0, 4.0) == 0.0
    end

    @testset "an offset array with axes 0:N matches the plain array with offset 1" begin
        N = 30
        u_plain = zeros(N + 1)
        u_plain[1] = 1.0
        u_offset = OffsetArray(zeros(N + 1), 0:N)
        u_offset[0] = 1.0

        sys_plain = FSPSystem(rs, OrderedIndexHandler(rs; offset = 1))
        sys_offset = FSPSystem(rs, OrderedIndexHandler(rs; offset = 0))
        prob_plain = ODEProblem(sys_plain, u_plain, (0.0, 1.0), pmap)
        prob_offset = ODEProblem(sys_offset, u_offset, (0.0, 1.0), pmap)

        du_plain = similar(u_plain)
        du_offset = similar(u_offset)
        prob_plain.f(du_plain, u_plain, prob_plain.p, 0.0)
        prob_offset.f(du_offset, u_offset, prob_offset.p, 0.0)
        @test collect(du_offset) ≈ du_plain
    end

    @testset "a shifted offset gives the same right-hand side on the shifted array" begin
        N = 30
        u_plain = zeros(N + 1)
        u_plain[1] = 1.0
        # Index k + 4 is count k, so the array axes are 4:(N + 4).
        u_shifted = OffsetArray(zeros(N + 1), 4:(N + 4))
        u_shifted[4] = 1.0

        sys_plain = FSPSystem(rs, OrderedIndexHandler(rs; offset = 1))
        sys_shifted = FSPSystem(rs, OrderedIndexHandler(rs; offset = 4))
        prob_plain = ODEProblem(sys_plain, u_plain, (0.0, 1.0), pmap)
        prob_shifted = ODEProblem(sys_shifted, u_shifted, (0.0, 1.0), pmap)

        du_plain = similar(u_plain)
        du_shifted = similar(u_shifted)
        prob_plain.f(du_plain, u_plain, prob_plain.p, 0.0)
        prob_shifted.f(du_shifted, u_shifted, prob_shifted.p, 0.0)
        @test collect(du_shifted) ≈ du_plain
    end
end

@testset "Per-dimension offsets follow the species order" begin
    rs = @reaction_network begin
        birth, 0 --> A
        death, A --> 0
        kb, 0 --> B
        db, B --> 0
    end

    # Dimension 1 tracks B with offset 0, dimension 2 tracks A with offset 2.
    ih = OrderedIndexHandler(rs, [:B, :A]; offset = [0, 2])
    sys = FSPSystem(rs, ih)

    # Index (5, 3) is A = 3 - 2 = 1 and B = 5. The death propensity of A is death * A = 4.
    @test sys.rfs[2]([5, 3], 0.0, 3.0, 4.0, 5.0, 6.0) == 4.0
    # If the offsets were applied to the wrong dimensions, A would read as 5 and the rate would be 20.
    @test sys.rfs[2]([5, 3], 0.0, 3.0, 4.0, 5.0, 6.0) != 20.0
end

@testset "Offset matrix matches the ODE right-hand side" begin
    rs = @reaction_network begin
        birth, 0 --> A
        death, A --> 0
    end
    pmap = [:birth => 3.0, :death => 4.0]
    N = 20

    u_offset = OffsetArray(zeros(N + 1), 0:N)
    u_offset[0] = 1.0
    sys = FSPSystem(rs, OrderedIndexHandler(rs; offset = 0))

    A = SparseMatrixCSC(sys, axes(u_offset), pmap, 0.0)
    prob = ODEProblem(sys, u_offset, (0.0, 1.0), pmap)
    du = similar(u_offset)
    prob.f(du, u_offset, prob.p, 0.0)
    @test collect(du) ≈ A * collect(u_offset)

    # The steady-state matrix conserves probability on the same axes.
    A_ss = SparseMatrixCSC(sys, axes(u_offset), pmap, SteadyState())
    @test all(abs.(sum(A_ss; dims = 1)) .< 1.0e-12)
end
