function sincos_approx(x, j = 1; L = 20, init = :taylor)
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return [one(T); x]
        elseif init == :interpolation
            xL = T(pi)/2^L
            return [1 + (cos(xL)-1)*x/xL; sin(xL)*x/xL]
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = T(pi)/2^j
    z = sincos_approx(pivot - abs(x-pivot), j+1; L, init)
    if x > pivot
        [cos(2*pivot) sin(2*pivot); sin(2*pivot) -cos(2*pivot)] * z
    else
        z
    end
end

function sincos_approx_relu(x, j = 1; L = 20, init = :taylor)
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return [one(T); x]
        elseif init == :interpolation
            xL = T(pi)/2^L
            return [1 + (cos(xL)-1)*x/xL; sin(xL)*x/xL]
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = T(pi)/2^j
    hx = pivot - relu(x-pivot) - relu(pivot-x)
    z = sincos_approx_relu(hx, j+1; L, init)
    a = x - pivot
    bhat1 = cos(2*pivot)*z[1] + sin(2*pivot)*z[2] - cos(pivot)
    chat1 = z[1] - cos(pivot)
    bhat2 = sin(2*pivot)*z[1] - cos(2*pivot)*z[2] - sin(pivot)
    chat2 = z[2] - sin(pivot)

    u1 = cos(pivot) + relu(a) - relu(-a) - relu(-bhat1 + a) + relu(chat1 - a)
    v1 = sin(pivot) + relu(a) - relu(-a) - relu(-bhat2 + a) + relu(chat2 - a)

    [u1; v1]
end

"""
    ann_sincos(L[, T]; init = :interpolation)

Construct a neural network to approximate `cos(x)` and `sin(x)` on
the interval `[0,1]`. The network implements a recursive algorithm with `L` steps.

Optionally, a numerical precision type `T` can be specified.

A keyword argument `init` can be specified with the values:
- 'taylor': the recursive algorithm is initialized with a Taylor series approximation
- 'interpolation': the algorithm constructs an interpolating approximation
"""
function ann_sincos(L, ::Type{T} = Float64; init = :interpolation) where {T}
    # Refinement: [x] -> [refine(x); x]
    Mref = ann_refine(L, T; scalefactor = T(pi))
    # Start of recursion:
    A0 = zeros(T, L+2, L+1)
    b0 = zeros(T, L+2)
    if init == :taylor
        A0[1,1] = 0
        A0[2,1] = 1
        b0[1] = 1
        A0[3:end,2:end] .= I(L)
        Minit = ReLUNet([A0],[b0])
    elseif init == :interpolation
        xL = T(pi)/2^L
        A0[1,1] = (cos(xL) - 1) / xL
        A0[2,1] = sin(xL) / xL
        b0[1] = 1
        A0[3:end,2:end] .= I(L)
        Minit = ReLUNet([A0],[b0])
    else
        throw(ArgumentError("Unknown initialization: :$(init)"))
    end
    M = hcat_model(Mref, Minit)
    # Recursion
    for j in L:-1:1
        M = hcat_model_skipremainder(M, ann_sincos_rec(j, T))
    end
    M
end

"Reduce [u_j,v_j,x_j] to [u_{j-1},v_{j-1}]."
function ann_sincos_rec(j, ::Type{T} = Float64) where {T}
    pivot = T(pi)/2^j

    # First, map from z[1], z[2] and x to [a, bhat1, chat1, bhat2, chat2]
    A0 = [0 0 1; cos(2*pivot) sin(2*pivot) 0; 1 0 0; sin(2*pivot) -cos(2*pivot) 0; 0 1 0]
    b0 = [-pivot; -cos(pivot); -cos(pivot); -sin(pivot); -sin(pivot)]
    M0 = ReLUNet([A0],[b0])

    # Then map to the inputs: [a, -a, -bhat1+a, chat1-a, -bhat2+a, chat2-a]
    A1 = [1 0 0 0 0; -1 0 0 0 0; 1 -1 0 0 0; -1 0 1 0 0; 1 0 0 -1 0; -1 0 0 0 1]
    b1 = zeros(T, 6)

    # Now collect the outcomes
    A2 = [1 -1 -1 1 0 0; 1 -1 0 0 -1 1]
    b2 = [cos(pivot); sin(pivot)]
    M1 = ReLUNet([A1,A2],[b1,b2])

    hcat_model(M0, M1)
end
