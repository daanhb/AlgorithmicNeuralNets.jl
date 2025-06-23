# AlgorithmicNeuralNets

[![Build Status](https://github.com/daanhb/AlgorithmicNeuralNets.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/daanhb/AlgorithmicNeuralNets.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/daanhb/AlgorithmicNeuralNets.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/daanhb/AlgorithmicNeuralNets.jl)

A package for the construction of neural networks from algorithms.

This code accompanies the paper "On the algorithmic construction of deep ReLU networks". In this README we show how to use the package to reproduce the results of the paper.

## A sorting algorithm using neural networks

The package implements the bitonic sort algorithm using neural networks. The following example constructs a neural network that sorts an input vector of length 1024.

```julia
julia> using AlgorithmicNeuralNets, LinearAlgebra

julia> M = bitonic_sort_network(10);

julia> size(M)
(1024, 1024)

julia> depth(M)
57

julia> width(M)
2048

julia> x = rand(2^10);

julia> norm(M(x)-sort(x))
2.2176687547934757e-15
```
The network has 57 layers. Its width is twice the dimension of the input vector. The final statement demonstrates that the sorting operation is exact, not approximate.

## Approximation of special functions

The paper describes neural networks that approximate a number of special functions. In each case the convergence rate of the approximation is exponential in the depth of the network.

### The square function

The best known example is the function `f(x) = x^2`. This construction is often used in the analysis of the approximation power of neural networks.
```julia
julia> M = ann_square(10);

julia> size(M)
(1, 1)

julia> depth(M)
12

julia> width(M)
4

julia> x = rand()
0.519732898303821

julia> abs(M(x)-x^2)
4.066203840302762e-8

julia> M = ann_square(24);

julia> abs(M(x)-x^2)
2.886579864025407e-15
```
This network is a special case. Its width is `4`, regardless of the depth of the network.

### The exponential function

A network is described in the paper that approximates both `exp(x)` and `exp(-x)` simultaneously. In this case, the width of the network is proportional to the logarithm of its depth.

```julia
julia> M = ann_exp(10);

julia> size(M)
(2, 1)

julia> depth(M)
22

julia> width(M)
24

julia> x = rand()
0.037070301507582504

julia> norm(M([x]) - [exp(x); exp(-x)])
1.0810517294049192e-9
```
Note that using `M([x])` we apply the network to the input vector `[x]`, rather than to the scalar number `x`.


