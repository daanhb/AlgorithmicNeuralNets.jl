
struct NetPlan{T}
    results ::  Vector{Vector{T}}
end

function netplan(m::ReLUNet{T}) where T
    results = [zeros(T, length(b)) for b in m.biases]
    NetPlan(results)
end

function relu!(x::AbstractVector)
    for i in 1:length(x)
        x[i] = relu(x[i])
    end
    x
end

function apply!(y::AbstractVector{T}, m::ReLUNet{T}, x::AbstractVector{T}, p::NetPlan{T}) where T
    @assert length(p.results) == depth(m)-1
    p.results[1] .= m.biases[1]
    mul!(p.results[1], m.weights[1], x, 1, 1)
    for i in 2:length(m.weights)
        relu!(p.results[i-1])
        p.results[i] .= m.biases[i]
        mul!(p.results[i], m.weights[i], p.results[i-1], 1, 1)
    end
    y .= p.results[end]
end
