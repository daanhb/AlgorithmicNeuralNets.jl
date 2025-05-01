module AlgorithmicNeuralNets

using LinearAlgebra
using SparseArrays

export ReLUNet,
    depth,
    width,
    nb_parameters,
    netplan,
    hcat_model,
    vcat_model,
    ann_id,
    ann_abs,
    ann_shift,
    ann_min2,
    ann_max2,
    ann_sort2,
    ann_square,
    ann_exp,
    ann_sincos,
    ann_monomials,
    ann_mul,
    bitonic_sort_network

relu(x::Number) = x >= 0 ? x : zero(x)
relu(x::AbstractVector) = relu.(x)

# Definition of ReLUNet
include("network.jl")
include("plan.jl")
include("calculus.jl")
include("basicnets.jl")

# Approximations
include("refinement.jl")
include("approx/square.jl")
include("approx/exp.jl")
include("approx/trig.jl")
include("approx/monomials.jl")
include("approx/multiplication.jl")

# Applications
include("applications/sort.jl")
include("applications/disk.jl")

function spline_interpolation(nodes, vals)
    @assert length(nodes) == length(vals)

    n = length(nodes)
    neurons = [n,1]
    b1 = -nodes
    A1 = ones(n,1)

    b2 = vals[1:1]
    A2 = ones(1,n)
    A2[1] = (vals[2]-vals[1])/(nodes[2]-nodes[1])
    wsum = A2[1]
    for i in 2:n-1
        z = (vals[i+1]-vals[i]) / (nodes[i+1]-nodes[i]) - wsum
        A2[i] = z
        wsum += z
    end
    ReLUNet([A1,A2], [b1,b2])
end

end
