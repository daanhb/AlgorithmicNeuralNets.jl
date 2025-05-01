# The calculus of ReLU networks

function hcat_model(m1::ReLUNet{T}, m2::ReLUNet{T}) where T
    @assert size(m2,2) == size(m1,1)
    weights = vcat(m1.weights[1:end-1], [m2.weights[1]*m1.weights[end]], m2.weights[2:end])
    biases = vcat(m1.biases[1:end-1], [m2.biases[1]+m2.weights[1]*m1.biases[end]], m2.biases[2:end])
    ReLUNet(weights, biases)
end

function hcat_model(m1::ReLUNet{S}, m2::ReLUNet{T}) where {S,T}
    U = promote_type(S,T)
    hcat_model(ReLUNet{U}(m1), ReLUNet{U}(m2))
end

hcat_model(m1::ReLUNet, m2::ReLUNet, m3::ReLUNet...) =
    hcat_model(hcat_model(m1, m2), m3...)

function vcat_model(m1::ReLUNet{T}, m2::ReLUNet{T}) where T
    @assert depth(m1) == depth(m2)
    weights = [hvcat((2,2), m1.weights[i], zeros(size(m1.weights[i],1),size(m2.weights[i],2)), zeros(size(m2.weights[i],1),size(m1.weights[i],2)), m2.weights[i]) for i in 1:length(m1.weights)]
    biases = map(vcat, m1.biases, m2.biases)
    ReLUNet(weights, biases)
end

function vcat_model(m1::ReLUNet{S}, m2::ReLUNet{T}) where {S,T}
    U = promote_type(S,T)
    vcat_model(ReLUNet{U}(m1), ReLUNet{U}(m2))
end

vcat_model(m1::ReLUNet, m2::ReLUNet, m3::ReLUNet...) =
    vcat_model(vcat_model(m1, m2), m3...)

function hcat_model_skipremainder(m1::ReLUNet{T}, m2::ReLUNet{T}) where T
    s1,t1 = size(m1)
    s2,t2 = size(m2)
    if s1 == t2
        return hcat_model(m1, m2)
    end
    @assert s1 > t2
    nskip = s1-t2
    m3 = ann_id(nskip, depth(m2), T)
    hcat_model(m1, vcat_model(m2, m3))
end

perturb_vecormat(A, eps) = A + eps*rand(eltype(A), size(A))

function perturb_model(m::ReLUNet, eps)
    weights = [perturb_vecormat(A, eps) for A in m.weights]
    biases = [perturb_vecormat(b, eps) for b in m.biases]
    ReLUNet(weights, biases)
end
