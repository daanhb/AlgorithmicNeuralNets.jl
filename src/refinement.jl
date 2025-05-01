
"""
    ann_refine(L[, T; scalefactor = 1])

Construct a neural network that will refine a scalar input `L` times, with
geometrically graded pivots on the interval `[0,1]`. Optionally, the interval
can be scaled with a scalefactor.
"""
function ann_refine(L, ::Type{T} = Float64; scalefactor = one(T)) where {T}
    @assert L > 0
    models = [ann_refine_rec(j, T; scalefactor) for j in 1:L]
    reduce(hcat_model_skipremainder, models)
end

"Map `x` to `[s - abs(x-s), x]` where `s = 2^(-j)*scalefactor`."
function ann_refine_rec(j, ::Type{T} = Float64; scalefactor = one(T)) where {T}
    pivot = one(T)/2^j * scalefactor
    A1 = zeros(T, 4, 1)
    A1 .= [1; -1; 1; -1]
    b1 = [-pivot; pivot; 0; 0]
    A2 = T[-1 -1 0 0; 0 0 1 -1]
    b2 = T[pivot; 0]
    ReLUNet([A1,A2],[b1,b2])
end

function ann_refine_2d(L, ::Type{T} = Float64; scalefactor = one(T)) where {T}
    @assert L > 0
    models = [ann_refine_2d_rec(j, T; scalefactor) for j in 1:L]
    reduce(hcat_model_skipremainder, models)
end

"Map `[x,y]` to `[s - abs(x-s), s-abs(y-s), x, y]` where `s = 2^(-j)*scalefactor`."
function ann_refine_2d_rec(j, ::Type{T} = Float64; scalefactor = one(T)) where {T}
    pivot = one(T)/2^j * scalefactor
    A1 = T[1 0; -1 0; 1 0; -1 0; 0 1; 0 -1; 0 1; 0 -1]
    b1 = [-pivot; pivot; 0; 0; -pivot; pivot; 0; 0]
    A2 = T[-1 -1 0 0 0 0 0 0; 0 0 0 0 -1 -1 0 0; 0 0 1 -1 0 0 0 0; 0 0 0 0 0 0 1 -1]
    b2 = T[pivot; pivot; 0; 0]
    ReLUNet([A1,A2],[b1,b2])
end
