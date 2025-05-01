
abstract type NeuralNetwork end

"""
A ReLU neural network.
"""
struct ReLUNet{T,AA <: AbstractMatrix{T}} <: NeuralNetwork
    weights     ::  Vector{AA}
    biases      ::  Vector{Vector{T}}

    function ReLUNet{T,AA}(weights::Vector{<:AbstractMatrix}, biases::Vector{<:Vector}) where {T,AA<:AbstractMatrix{T}}
        # do sanity checks
        @assert length(weights) == length(biases)
        for i in 1:length(weights)
            @assert length(biases[i]) == size(weights[i],1)
        end
        new(weights, biases)
    end
end

const DenseReLUNet{T} = ReLUNet{T,Matrix{T}}
const SparseReLUNet{T} = ReLUNet{T,SparseMatrixCSC{T,Int}}

ReLUNet(weights::Vector{<:AbstractMatrix{T}}) where {T} = ReLUNet{T}(weights)

function ReLUNet{T}(weights::Vector{<:AbstractMatrix}) where {T}
    biases = [zeros(T,size(weights[i],1)) for i in 1:length(weights)]
    ReLUNet{T}(weights, biases)
end

ReLUNet(weights::Vector{AA}, biases::Vector{Vector{T}}) where {S,T,AA<:AbstractMatrix{S}} =
    ReLUNet{promote_type(S,T)}(weights, biases)

ReLUNet{T}(weights::Vector{Matrix{T}}, biases::Vector{Vector{T}}) where {T} =
    ReLUNet{T,Matrix{T}}(weights, biases)
ReLUNet{T}(weights::Vector{<:Matrix}, biases::Vector{<:Vector}) where {T} =
    ReLUNet{T,Matrix{T}}(weights, biases)
ReLUNet{T}(weights::Vector{AA}, biases::Vector{Vector{T}}) where {T,AA<:AbstractMatrix{T}} =
    ReLUNet{T,AA}(weights, biases)

ReLUNet{T}(m::ReLUNet) where T = ReLUNet{T}(m.weights, m.biases)

SparseReLUNet(m::DenseReLUNet{T}) where T = SparseReLUNet{T}(m)
function SparseReLUNet{T}(m::DenseReLUNet{T}) where T
    weights = [sparse(A) for A in m.weights]
    biases = [b for b in m.biases]
    SparseReLUNet{T}(weights, biases)
end

depth(m::ReLUNet) = length(m.weights)+1
width(m::ReLUNet) = maximum(map(length, m.biases))

function nb_parameters(m::ReLUNet)
    n1 = sum(map(A->prod(size(A)), m.weights))
    n2 = sum(map(length, m.biases))
    n1+n2
end
function nnz_parameters(m::ReLUNet)
    n1 = sum(map(nnz, m.weights))
    n1
    # n2 = sum(map(nnz, m.biases))
    # n1+n2
end

Base.size(m::ReLUNet) = (size(m.weights[end],1), size(m.weights[1],2))
function Base.size(m::ReLUNet, i::Int)
    if i == 1
        size(m.weights[end],1)
    elseif i == 2
        size(m.weights[1],2)
    else
        1
    end
end

"Return the i-th weight of the net (without copy of the data)."
weight(m::ReLUNet, i::Int) = m.weights[i]

"Return the i-th bias of the net (without copy of the data)."
bias(m::ReLUNet, i::Int) = m.biases[i]

(m::ReLUNet)(x) = apply(m, x)
(m::ReLUNet)(x::Number) = apply(m, x)[1]

function apply(m::ReLUNet, x; verbose=false)
    @assert length(x) == size(m)[2]

    y = x
    if verbose
        println("Network input: ", y)
    end
    for i in 1:depth(m)-2
        y = m.weights[i]*y + m.biases[i]
        if verbose
            println("Layer $(i) input: ", y)
        end
        y = relu.(y)
    end
    y = m.weights[end]*y + m.biases[end]
end

function ReLUNet{T}(neurons::Vector{Int}) where T
    weights = Matrix{T}[]
    for i in 1:length(neurons)-1
        push!(weights, zeros(T, neurons[i+1], neurons[i]))
    end
    ReLUNet{T}(weights)
end
