using Test
using Catalyst: @reaction_network, species
using FiniteStateProjection: AbstractIndexHandler, FSPSystem, OrderedIndexHandler
import FiniteStateProjection:
    build_rhs_header, getsubstitutions, pairedindices, singleindices
using SciMLBase: ODEProblem
import Base: LinearIndices, vec
import SparseArrays: SparseMatrixCSC

struct DelegatingIndexHandler{N} <: AbstractIndexHandler
    delegate::OrderedIndexHandler{N}
end

singleindices(ih::DelegatingIndexHandler, arr::AbstractArray) = singleindices(ih.delegate, arr)
singleindices(ih::DelegatingIndexHandler, dims::Tuple) = singleindices(ih.delegate, dims)
pairedindices(ih::DelegatingIndexHandler, arr::AbstractArray, shift::CartesianIndex) =
    pairedindices(ih.delegate, arr, shift)
pairedindices(ih::DelegatingIndexHandler, dims::Tuple, shift::CartesianIndex) =
    pairedindices(ih.delegate, dims, shift)
getsubstitutions(ih::DelegatingIndexHandler, rs; state_sym) =
    getsubstitutions(ih.delegate, rs; state_sym)
LinearIndices(ih::DelegatingIndexHandler, arr) = LinearIndices(ih.delegate, arr)
vec(ih::DelegatingIndexHandler, arr) = vec(ih.delegate, arr)

function build_rhs_header(::FSPSystem{<:DelegatingIndexHandler})
    return quote
        birth, death = p
    end
end

@testset "Public index-handler interface" begin
    rs = @reaction_network begin
        birth, 0 --> X
        death, X --> 0
    end
    pmap = [:birth => 2.0, :death => 1.0]
    u0 = [1.0, 0.0, 0.0]

    default_system = FSPSystem(rs)
    custom_system = FSPSystem(rs, DelegatingIndexHandler(OrderedIndexHandler(rs, [:X])))

    default_matrix = SparseMatrixCSC(default_system, size(u0), pmap, 0.0)
    custom_matrix = SparseMatrixCSC(custom_system, size(u0), pmap, 0.0)
    @test custom_matrix == default_matrix

    default_problem = ODEProblem(default_system, u0, (0.0, 1.0), pmap)
    custom_problem = ODEProblem(custom_system, u0, (0.0, 1.0), pmap)
    default_du = similar(u0)
    custom_du = similar(u0)
    default_problem.f(default_du, u0, default_problem.p, 0.0)
    custom_problem.f(custom_du, u0, custom_problem.p, 0.0)
    @test custom_du == default_du
end

@testset "OrderedIndexHandler construction" begin
    rs = @reaction_network begin
        k1, 0 --> A
        k2, A --> 0
        k3, 0 --> B
    end

    @testset "without an order, keeps the Catalyst species order" begin
        @test OrderedIndexHandler(rs).perm == (1, 2)
        @test FSPSystem(rs).ih == OrderedIndexHandler(rs)
    end

    @testset "with an order of species names" begin
        ih = OrderedIndexHandler(rs, [:B, :A])
        @test ih.perm == (2, 1)
        @test FSPSystem(rs, [:B, :A]).ih == ih
    end

    @testset "with an order of Catalyst species" begin
        ih = OrderedIndexHandler(rs, reverse(species(rs)))
        @test ih.perm == (2, 1)
    end

    @testset "the zero-argument form is not defined" begin
        @test_throws MethodError OrderedIndexHandler{2}()
    end

    @testset "offset defaults to 1 for all species" begin
        @test OrderedIndexHandler(rs).offset == (1, 1)
        @test OrderedIndexHandler(rs, [:B, :A]).offset == (1, 1)
    end

    @testset "offset is a keyword argument" begin
        @test OrderedIndexHandler(rs; offset = 0).offset == (0, 0)
        @test OrderedIndexHandler(rs, [:A, :B]; offset = 0).offset == (0, 0)
        @test OrderedIndexHandler(rs; offset = [0, 2]).offset == (0, 2)
        # One offset per dimension, in the order given by `order`.
        ih = OrderedIndexHandler(rs, [:B, :A]; offset = [0, 2])
        @test ih.perm == (2, 1)
        @test ih.offset == (0, 2)
    end

    @testset "offset must have one entry per species" begin
        @test_throws DimensionMismatch OrderedIndexHandler(rs; offset = [0, 1, 2])
        @test_throws DimensionMismatch OrderedIndexHandler(rs, [:A, :B]; offset = [0])
    end

    @testset "FSPSystem passes the offset on" begin
        sys = FSPSystem(rs, [:A, :B]; offset = 0)
        @test sys.ih.offset == (0, 0)
    end
end

@testset "Index handler error handling" begin
    rs = @reaction_network begin
        k1, 0 --> A
        k2, A --> 0
        k3, 0 --> B
    end

    @testset "rejects a wrong number of species" begin
        @test_throws DimensionMismatch OrderedIndexHandler(rs, [:A])
    end

    @testset "rejects an unknown species" begin
        @test_throws ArgumentError OrderedIndexHandler(rs, [:A, :C])
    end

    @testset "rejects a repeated species" begin
        @test_throws ArgumentError OrderedIndexHandler(rs, [:A, :A])
    end

    @testset "pairedindices rejects a dimension tuple of the wrong length" begin
        ih = OrderedIndexHandler(rs, [:A, :B])
        @test_throws DimensionMismatch pairedindices(ih, (3,), CartesianIndex(1, 0))
        @test_throws DimensionMismatch pairedindices(ih, (3, 3, 3), CartesianIndex(1, 0))
    end

    @testset "pairedindices accepts a dimension tuple of the right length" begin
        ih = OrderedIndexHandler(rs, [:A, :B])
        @test length(collect(pairedindices(ih, (3, 3), CartesianIndex(1, 0)))) == 6
    end
end
