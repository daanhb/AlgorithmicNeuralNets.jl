function monomials_approx(x, degree, j = 1; L = 20, init = :interpolation)
    @assert degree >= 2
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return zeros(T, degree-1)
        elseif init == :interpolation
            xL = one(T)/2^L
            return [xL^k * x/xL for k in 2:degree]
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    z = monomials_approx(pivot - abs(x-pivot), degree, j+1; L, init)
    if x > pivot
        u = zeros(T, degree+1)
        u[1] = 1
        u[2] = x
        for k in 2:degree
            u[k+1] = (-1)^k*z[k-1]
            for l in 0:k-1
                u[k+1] = u[k+1] - binomial(k, l)*(-1)^(k+l)*2^(k-l)*pivot^(k-l) * u[l+1]
            end
        end
        z = u[3:end]
    else
        z
    end
end

function monomials_approx_relu(x, degree, j = 1; L = 20, init = :interpolation)
    @assert degree >= 2
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return zeros(T, degree-1)
        elseif init == :interpolation
            xL = one(T)/2^L
            return [xL^k * x/xL for k in 2:degree]
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    hx = pivot - relu(x-pivot) - relu(pivot-x)
    z = monomials_approx_relu(hx, degree, j+1; L, init)

    a = x - pivot
    bhat = zeros(T, degree-1)
    chat = zeros(T, degree-1)
    C = Matrix(I(degree+1) * one(T))
    for k in 2:degree
        Ck = Matrix(I(degree+1) * one(T))
        for l in 0:k
            Ck[k+1,l+1] = -binomial(k, l)*(-1)^(k-l)*(2pivot)^(k-l)
        end
        Ck[k+1,k+1] = (-1)^k
        C = Ck * C
        b = C * vcat([1; x], z)
        bhat[k-1] = b[k+1] - pivot^k
        chat[k-1] = z[k-1] - pivot^k
    end
    u = zeros(T, degree-1)
    beta = zeros(T, degree-1)
    for k in 2:degree
        beta[k-1] = k * pivot^(k-1)
        u[k-1] = pivot.^k + beta[k-1]*relu(a) - relu(-a) + relu(bhat[k-1] - beta[k-1]*a) + relu(chat[k-1] - a)
    end
    u
end

"""
    ann_monomials(L, degree[, T]; init = :interpolation)

Construct a neural network to approximate monomials up to the given degree on
the interval `[0,1]`. The network implements a recursive algorithm with `L` steps.

Optionally, a numerical precision type `T` can be specified.

A keyword argument `init` can be specified with the values:
- 'taylor': the recursive algorithm is initialized with a Taylor series approximation
- 'interpolation': the algorithm constructs an interpolating approximation
"""
function ann_monomials(L, degree::Int, ::Type{T} = Float64; init = :interpolation) where {T}
    # Refinement: [x] -> [refine(x); x]
    Mref = ann_refine(L, T)
    # Start of recursion:
    A0 = zeros(T, L+degree-1, L+1)
    if init == :taylor
        A0[degree:end,2:end] .= I(L)
    elseif init == :interpolation
        xL = one(T)/2^L
        for k in 2:degree
            A0[k-1,1] = xL^(k-1)
        end
        A0[degree:end,2:end] .= I(L)
    else
        throw(ArgumentError("Unknown initialization: :$(init)"))
    end
    Minit = ReLUNet([A0])

    M = hcat_model(Mref, Minit)
    # Recursion
    for j in L:-1:1
        M = hcat_model_skipremainder(M, ann_monomials_rec(j, degree, T))
    end
    M
end

function ann_monomials_rec(j, degree, ::Type{T} = Float64) where T
    pivot = one(T)/2^j

    # First, map from z[1:degree-1] and x to [a, bhat1, chat1, ...]
    A0 = zeros(T, 1+2*(degree-1), degree)
    b0 = zeros(T, 1+2*(degree-1))
    # a = x-pivot
    A0[1,degree] = 1
    b0[1] = -pivot
    # For bhat and chat we need to construct the recurrence relation.
    C = Matrix(I(degree+1) * one(T))
    for k in 2:degree
        Ck = Matrix(I(degree+1) * one(T))
        for l in 0:k
            Ck[k+1,l+1] = -binomial(k, l)*(-1)^(k-l)*(2pivot)^(k-l)
        end
        Ck[k+1,k+1] = (-1)^k
        C = Ck * C
        # Compute bhat[k-1]
        # Here, we need row k+1 of C. Matrix C multiplies [1; x; Z], but here we have to consider 1 and x separately.
        A0[1+2*(k-2)+1,1:degree-1] = C[k+1,3:end]
        # -> the coefficient multiplying x is the second element of the row
        A0[1+2*(k-2)+1,degree] = C[k+1,2]
        # -> the constant term goes in the vector b
        b0[1+2*(k-2)+1] = C[k+1,1] - pivot^k
        # Compute chat[k-1]
        A0[1+2*(k-2)+2,k-1] = 1
        b0[1+2*(k-2)+2] = -pivot^k
    end
    M0 = ReLUNet([A0],[b0])

    beta = zeros(T, degree-1)
    for k in 2:degree
        beta[k-1] = k * pivot^(k-1)
    end

    # Next, we map to the inputs of ReLU: [a, bhat1, chat1, ...] -> [a, -a, bhat1-beta1*a, chat1-a, ...]
    A1 = zeros(T, 2+2*(degree-1), 1+2*(degree-1))
    b1 = zeros(T, 2+2*(degree-1))
    A1[1,1] = 1
    A1[2,1] = -1
    for k in 2:degree
        A1[2+2*(k-2)+1,1] = -beta[k-1]
        A1[2+2*(k-2)+1,1+2*(k-2)+1] = 1
        A1[2+2*(k-2)+2,1] = -1
        A1[2+2*(k-2)+2,1+2*(k-2)+2] = 1
    end

    # Finally we map to the outcomes
    A2 = zeros(T, degree-1, 2+2*(degree-1))
    b2 = zeros(T, degree-1)
    for k in 2:degree
        b2[k-1] = pivot^k
        A2[k-1,1] = beta[k-1]
        A2[k-1,2] = -1
        A2[k-1,2+2*(k-2)+1] = 1
        A2[k-1,2+2*(k-2)+2] = 1
    end
    M1 = ReLUNet([A1,A2],[b1,b2])

    hcat_model(M0, M1)
end
