using Test
using AxisArrays: AxisArray, Axis
using Catalyst: @reaction_network
using FiniteStateProjection
using SciMLBase: ODEProblem, SteadyStateProblem

@testset "Labelled state arrays" begin
    rs = @reaction_network begin
        s1, 0 --> A
        d1, A --> 0
        s2, 0 --> B
        d2, B --> 0
    end
    pmap = [:s1 => 1.0, :d1 => 1.0, :s2 => 2.0, :d2 => 0.5]

    sys = FSPSystem(rs, [:A, :B])
    sys_perm = FSPSystem(rs, [:B, :A])

    @testset "the species order sets the state-array dimensions" begin
        @test sys.ih.perm == (1, 2)
        @test sys_perm.ih.perm == (2, 1)
    end

    u = rand(5, 4)
    u ./= sum(u)

    # The same distribution with axes labelled by species, in both orders.
    u_ab = AxisArray(u, Axis{:A}(0:4), Axis{:B}(0:3))
    u_ba = AxisArray(permutedims(u), Axis{:B}(0:3), Axis{:A}(0:4))

    @testset "plain arrays are unchanged" begin
        prob = ODEProblem(sys, u, (0.0, 1.0), pmap)
        @test prob.u0 == u
    end

    @testset "labelled arrays are reordered to the system order" begin
        @test ODEProblem(sys, u_ab, (0.0, 1.0), pmap).u0 ≈ u
        @test ODEProblem(sys, u_ba, (0.0, 1.0), pmap).u0 ≈ u
        @test SteadyStateProblem(sys, u_ab, pmap).u0 ≈ u
        @test SteadyStateProblem(sys, u_ba, pmap).u0 ≈ u
    end

    @testset "labelled arrays are reordered for a custom species order" begin
        @test ODEProblem(sys_perm, u_ab, (0.0, 1.0), pmap).u0 ≈ permutedims(u)
        @test ODEProblem(sys_perm, u_ba, (0.0, 1.0), pmap).u0 ≈ permutedims(u)
    end

    @testset "axis names must match the species" begin
        wrong = AxisArray(u, Axis{:A}(0:4), Axis{:C}(0:3))
        @test_throws ArgumentError ODEProblem(sys, wrong, (0.0, 1.0), pmap)
        @test_throws ArgumentError SteadyStateProblem(sys, wrong, pmap)

        too_many = AxisArray(rand(5, 4, 2), Axis{:A}(0:4), Axis{:B}(0:3), Axis{:C}(0:1))
        @test_throws ArgumentError ODEProblem(sys, too_many, (0.0, 1.0), pmap)
    end

    @testset "system built from a template" begin
        template = AxisArray(zeros(4, 5), Axis{:B}(0:3), Axis{:A}(0:4))
        sys_t = FSPSystem(rs, template)
        @test sys_t.ih.perm == (2, 1)
        @test sys_t.ih == OrderedIndexHandler(rs, [:B, :A])
        @test sys_t.ih.offset == (1, 1)
    end

    @testset "offsets default to 1 whatever the axis values" begin
        template = AxisArray(zeros(5, 4), Axis{:A}(2:6), Axis{:B}(0:3))
        sys_t = FSPSystem(rs, template)
        @test sys_t.ih.perm == (1, 2)
        @test sys_t.ih.offset == (1, 1)
        @test sys_t.ih == OrderedIndexHandler(rs, [:A, :B])
    end

    @testset "a one-dimensional template" begin
        rs1 = @reaction_network begin
            s, 0 --> A
            d, A --> 0
        end
        sys1 = FSPSystem(rs1, AxisArray(zeros(4), Axis{:A}(0:3)))
        @test sys1.ih.perm == (1,)
        @test sys1.ih.offset == (1,)
    end

    @testset "offsets can be overridden by species" begin
        template = AxisArray(zeros(5, 4), Axis{:A}(0:4), Axis{:B}(0:3))
        @test FSPSystem(rs, template; offset = [:A => 2]).ih.offset == (2, 1)
        # A species that is not listed keeps the default.
        @test FSPSystem(rs, template; offset = [:B => 0]).ih.offset == (1, 0)
        @test FSPSystem(rs, template; offset = [:A => 2, :B => 3]).ih.offset == (2, 3)
    end

    @testset "offset overrides must name axes and be integers" begin
        template = AxisArray(zeros(5, 4), Axis{:A}(0:4), Axis{:B}(0:3))
        @test_throws ArgumentError FSPSystem(rs, template; offset = [:C => 2])
        @test_throws ArgumentError FSPSystem(rs, template; offset = [:A => 1.5])
    end

    @testset "axis values are labels and do not set offsets" begin
        rs1 = @reaction_network begin
            s, 0 --> A
            d, A --> 0
        end
        labels = AxisArray(zeros(4), Axis{:A}(["a", "b", "c", "d"]))
        @test FSPSystem(rs1, labels).ih.offset == (1,)
    end

    @testset "template species must belong to the network" begin
        template = AxisArray(zeros(5, 4), Axis{:A}(0:4), Axis{:C}(0:3))
        @test_throws ArgumentError FSPSystem(rs, template)
    end
end
