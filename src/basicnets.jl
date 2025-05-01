
"CRN for x -> x."
function ann_id(::Type{T} = Float64) where T
    A1 = zeros(T, 2, 1)
    A1 .= [1; -1]
    A2 = T[1 -1]
    ReLUNet([A1,A2])
end

"CRN for x -> 0."
function ann_zero(::Type{T} = Float64) where T
    A1 = zeros(T, 1, 1)
    ReLUNet([A1,A1])
end

"CRN for x -> C."
function ann_constant(C::T) where {T}
    A1 = zeros(T, 1, 1)
    b1 = zeros(T, 1)
    b2 = [C]
    ReLUNet([A1,A1],[b1,b2])
end

"Select the i-th output from a network of size n."
function ann_select(i::Int, n::Int, ::Type{T} = Float64) where {T}
    A = zeros(T, 1, n)
    A[i] = 1
    ReLUNet([A])
end

"Provide a single input as element i of an n-dimensional input."
function ann_input(i::Int, n::Int, ::Type{T} = Float64) where {T}
    A = zeros(T, n, 1)
    A[i] = 1
    ReLUNet([A])
end

"The weight matrices of an n-dimensional skip connection."
function skip_matrices(n, ::Type{T} = Float64) where T
    A0 = zeros(T, 2n, n)
    A1 = zeros(T, n, 2n)
    for k in 1:n
        A0[2*(k-1)+1,k] = 1
        A0[2*(k-1)+2,k] = -1
    end
    A1 = collect(A0')
    A0, A1
end

"CRN of the given depth for the map [x] -> [x] where [x] is n-dimensional."
function ann_id(n::Int, depth::Int = 3, ::Type{T} = Float64) where T
    @assert depth > 2
    A0, A1 = skip_matrices(n, T)
    M0 = ReLUNet([A0,A1])
    reduce(hcat_model, [M0 for i in 1:depth-2])
end


"Two-layer CRN (without ReLU) for x -> x+b."
function ann_shift(b::T) where {T<:Number}
    A1 = zeros(T, 1, 1)
    A1[1] = 1
    b1 = [b]
    ReLUNet([A1],[b1])
end

"CRN for x -> abs(x)."
function ann_abs(::Type{T} = Float64) where T
    A1 = zeros(T, 2, 1)
    A1 .= [1; -1]
    A2 = T[1 1]
    ReLUNet([A1,A2])
end

"CRN for (x,y) -> min(x,y)."
function ann_min2(::Type{T} = Float64) where T
    A1 = T[-1 1; 0 1; 0 -1]
    A2 = T[-1 1 -1]
    ReLUNet([A1,A2])
end

"CRN for (x,y) -> max(x,y)."
function ann_max2(::Type{T} = Float64) where T
    A1 = T[1 -1; 0 1; 0 -1]
    A2 = T[1 1 -1]
    ReLUNet([A1,A2])
end

"CRN for (x,y) -> [min(x,y),max(x,y)]."
function ann_sort2(::Type{T} = Float64) where T
    A1 = T[1 -1; -1 1; 0 1; 0 -1]
    A2 = T[0 -1 1 -1; 1 0 1 -1]
    ReLUNet([A1,A2])
end

"CRN for (x,y) -> (y,x)."
function ann_permute2(::Type{T} = Float64) where T
    A1 = T[0 1; 1 0]
    ReLUNet([A1])
end
