using AlgorithmicNeuralNets
using LinearAlgebra
using Test

const AN = AlgorithmicNeuralNets

include("test_basicnets.jl")
include("test_calculus.jl")
include("test_approximation.jl")


@testset "Applications: sorting" begin
    L = 8
    ntests = 10
    for i in 1:ntests
        x = rand(2^L)
        xs = sort(x)
        M = bitonic_sort_network(L)
        @test norm(xs-M(x)) < 1e-10
    end
end
