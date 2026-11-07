using Test
using Catalyst: @reaction_network
using FiniteStateProjection
using FiniteStateProjection: PermutingIndexHandler

@testset "Deprecated index-handler names" begin
    rs = @reaction_network begin
        k1, 0 --> A
        k2, A --> 0
        k3, 0 --> B
    end

    @testset "DefaultIndexHandler is an alias of OrderedIndexHandler" begin
        @test DefaultIndexHandler{2}(1, (1, 2)) isa OrderedIndexHandler{2}
        @test DefaultIndexHandler{2}(1, (1, 2)) == OrderedIndexHandler(rs, [:A, :B])
        @test DefaultIndexHandler{2}(1, (2, 1)) == OrderedIndexHandler(rs, [:B, :A])
    end

    @testset "PermutingIndexHandler forwards to OrderedIndexHandler" begin
        @test PermutingIndexHandler(rs, [:B, :A]) == OrderedIndexHandler(rs, [:B, :A])
    end

    @testset "NaiveIndexHandler forwards to OrderedIndexHandler" begin
        @test NaiveIndexHandler(1, (2, 1)) == OrderedIndexHandler(rs, [:B, :A])
    end
end
