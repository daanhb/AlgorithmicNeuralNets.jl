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
0.7627133997007148

julia> abs(M(x)-x^2)
9.186748993750271e-7

julia> M = ann_square(24);

julia> abs(M(x)-x^2)
9.992007221626409e-16
```
This network is a special case. Its width is `4`, regardless of the depth of the network.

### The exponential function

A network is described in the paper that approximates both `exp(x)` and `exp(-x)` simultaneously. In this case, the width of the network is proportional to its depth.

```julia
julia> M = ann_exp(10);

julia> size(M)
(2, 1)

julia> depth(M)
22

julia> width(M)
24

julia> x = rand()
0.312767561056721

julia> norm(M([x]) - [exp(x); exp(-x)])
1.470830763310804e-7

julia> M = ann_exp(24);

julia> depth(M)
50

julia> width(M)
52

julia> norm(M([x]) - [exp(x); exp(-x)])
1.1102230246251565e-16
```
Note that the syntax `M([x])` is used here to apply the network to the input vector `[x]`. The outcome is a vector, in this case of length 2.

### Trigonometric functions

The approximation of the `cos` and `sin` functions is similar to that of the exponential functions.
```julia
julia> M = ann_sincos(24);

julia> x = rand()
0.5830455566554003

julia> norm(M([x]) - [cos(x); sin(x)])
4.220308060520364e-15
```

### Monomials

A neural network to approximate the polynomial basis functions `x^k` is different from the earlier examples, since the dimension of the output grows with increasing
degree of the polynomials.

The degree is specified as the second argument. 
```julia
julia> M = ann_monomials(10, 5);

julia> depth(M)
22

julia> width(M)
28

julia> x = rand()
0.09437718256017558

julia> M([x])
4-element Vector{Float64}:
 0.008907271713001846
 0.0008406845084987791
 7.934727343766879e-5
 7.489305687780057e-6

julia> norm(M([x]) - x.^(2:5))
2.280291711793557e-7

julia> M = ann_monomials(24, 10);

julia> norm(M([x]) - x.^(2:10))
8.638733852373792e-16
```
The output omits the first two powers of `x` as these are `0` and `x` itself,
respectively.



## The multiplication function

The function `f(x,y) = xy` is another important known case. It can be used as a building block to approximate functions of several arguments, for example multivariate polynomials.

```julia
julia> M = ann_mul(24);

julia> depth(M)
74

julia> width(M)
102

julia> M([0.3,0.4])
1-element Vector{Float64}:
 0.11999999999999889
```


## Internal structure

The type `ReLUNet` represents a neural network. It is a very simple data container that holds the weights matrices and bias vectors. The biases are stored as a vector of vectors. The weights are stored as a vector of matrices.

For the case of the search algorithms, the weights matrices are stored as sparse matrices. In the other cases, the weight matrices are dense.
```julia
julia> M = bitonic_sort_network(10);

julia> typeof(M.biases)
Vector{Vector{Float64}} (alias for Array{Array{Float64, 1}, 1})

julia> typeof(M.weights)
Vector{SparseMatrixCSC{Float64, Int64}} (alias for Array{SparseArrays.SparseMatrixCSC{Float64, Int64}, 1})
```

