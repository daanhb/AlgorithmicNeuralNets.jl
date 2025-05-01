ann_foldline(a, b, c) = ann_foldline(promote(a, b, c)...)

"Fold the plane across the line a*x+b*y+c=0."
function ann_foldline(a::T, b::T, c::T) where T
    A1_abs = [a b; -a -b] / (a^2+b^2)
    b1_abs = [c; -c] / (a^2+b^2)
    A1_id = T[1 0; -1 0; 0 1; 0 -1]
    b1_id = T[0; 0; 0; 0]

    A1 = vcat(A1_abs, A1_id)
    b1 = vcat(b1_abs, b1_id)

    A2_x = T[0 0 1 -1 0 0]
    A2_y = T[0 0 0 0 1 -1]
    A2_abs = T[1 1 0 0 0 0]
    b2_abs = T[0]
    A2_proj = (a*A2_x + b*A2_y) / (a^2+b^2)
    b2_proj = [c/(a^2+b^2)]
    
    A2_xc = A2_x - a*A2_proj
    A2_yc = A2_y - b*A2_proj
    b2_xc = -a*b2_proj
    b2_yc = -b*b2_proj
    
    A2_xtilde = A2_xc - a*A2_abs
    A2_ytilde = A2_yc - b*A2_abs
    b2_xtilde = b2_xc - a*b2_abs
    b2_ytilde = b2_yc - b*b2_abs

    A2 = vcat(A2_xtilde, A2_ytilde)
    b2 = vcat(b2_xtilde, b2_ytilde)
    ReLUNet([A1,A2],[b1,b2])
end

"Compute the norm of a 2D vector."
function ann_vectornorm2(levels, ::Type{T} = Float64) where T
    theta = T(pi)
    foldmodels = [ann_foldline(-sin(theta/2^i), cos(theta/2^i), 0) for i in 0:levels]
    M = reduce(hcat_model, foldmodels)

    Mselect = ann_select(1, 2, T)
    hcat_model(M, Mselect)
end

function ann_disk(levels, radius::T = 1.0) where {T}
    M_norm = ann_vectornorm2(levels, T)

    A1 = zeros(T, 1, 1)
    A1[1,1] = -1
    A2 = zeros(T, 1, 1)
    A2[1,1] = 1
    M_radius = ReLUNet([A1,A2],[[radius],[0]])
    hcat_model(M_norm, M_radius)
end

"Compute the vector norm of a 4D vector."
function ann_vectornorm4(levels, ::Type{T} = Float64) where T
    M2 = ann_vectornorm2(levels, T)
    Ma = vcat_model(M2, M2)
    hcat_model(Ma, M2)
end

"Compute the vector norm of a 8D vector."
function ann_vectornorm8(levels, ::Type{T} = Float64) where T
    M4 = ann_vectornorm4(levels, T)
    M2 = ann_vectornorm2(levels, T)
    hcat_model(vcat_model(M4, M4), M2)
end
