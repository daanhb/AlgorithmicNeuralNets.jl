function exp_approx(x, j = 1; L = 20, init = :taylor)
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return [one(T) + x; one(T) - x]
        elseif init == :interpolation
            xL = one(T)/2^L
            return [1 + (exp(xL)-1)*x/xL; 1 + (exp(-xL)-1)*x/xL]
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    z = exp_approx(pivot - abs(x-pivot), j+1; L, init)
    if x > pivot
        [exp(2*pivot)*z[2]; exp(-2*pivot)*z[1]]
    else
        z
    end
end

# For documentation purposes
function exp_approx_relu(x, j = 1; L = 20, init = :taylor)
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return [one(T) + x; one(T) - x]
        elseif init == :interpolation
            xL = one(T)/2^L
            return [1 + (exp(xL)-1)*x/xL; 1 + (exp(-xL)-1)*x/xL]
        else
            throw(ArgumentError("Unknown or unsupported initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    hx = pivot - relu(x-pivot) - relu(pivot-x)
    z = exp_approx_relu(hx, j+1; L, init)

    beta1 = (exp(2*pivot)-exp(pivot)) / pivot
    gamma1 = exp(pivot)
    beta2 = exp(-pivot)
    gamma2 = (1-exp(-pivot))/pivot

    a = x - pivot
    bhat1 = exp(2*pivot)*z[2] - exp(pivot)
    chat1 = z[1] - exp(pivot)
    bhat2 = exp(-2*pivot)*z[1] - exp(-pivot)
    chat2 = z[2] - exp(-pivot)

    u1 = exp(pivot) + beta1*relu(a) - gamma1 * relu(-a) - relu(-bhat1 + beta1 * a) + relu(chat1 - gamma1 * a)
    v1 = exp(-pivot) + beta2*relu(a) - gamma2 * relu(-a) - relu(-bhat2 + beta2 * a) + relu(chat2 - gamma2 * a)
    [u1; v1]
end


function ann_exp(L, ::Type{T} = Float64; init=:taylor) where {T}
    # Refinement: [x] -> [refine(x); x]
    Mref = ann_refine(L, T)

    # Start of recursion: 
    A0 = zeros(T, L+2, L+1)
    b0 = zeros(T, L+2)
    if init == :taylor
        # We map [x_L, x_{L-1}, ..., x_1, x, y] -> [1+x_L, 1-x_L, x_{L-1}, x_{L-2}, ..., x]
        A0[1,1] = 1
        A0[2,1] = -1
        b0[1] = 1
        b0[2] = 1
        A0[3:end,2:end] .= I(L)
        Minit = ReLUNet([A0],[b0])
    elseif init == :interpolation
        xL = one(T)/2^L
        A0[1,1] = (exp(xL) - 1) / xL
        A0[2,1] = (exp(-xL) - 1) / xL
        b0[1] = 1
        b0[2] = 1
        A0[3:end,2:end] .= I(L)
        Minit = ReLUNet([A0],[b0])
    else
        throw(ArgumentError("Unknown initialization: :$(init)"))
    end
    M = hcat_model(Mref, Minit)

    # Recursive unfolding
    for j in L:-1:1
        M = hcat_model_skipremainder(M, ann_exp_rec(j, T))
    end
    M
end

"Reduce [u_j,v_j,x_j] to [u_{j-1},v_{j-1}]."
function ann_exp_rec(j, ::Type{T} = Float64) where {T}
    pivot = one(T)/2^j
    β1 = (exp(2*pivot)-exp(pivot)) / pivot
    γ1 = exp(pivot)
    β2 = exp(-pivot)
    γ2 = (1-exp(-pivot))/pivot

    # First, map from z[1], z[2] and x to [a, bhat1, chat1, bhat2, chat2]
    A0 = [0 0 1; 0 exp(2*pivot) 0; 1 0 0; exp(-2*pivot) 0 0; 0 1 0]
    b0 = [-pivot; -exp(pivot); -exp(pivot); -exp(-pivot); -exp(-pivot)]
    M0 = ReLUNet([A0],[b0])

    # Map to the inputs: [a, -a, -bhat1+β1*a, chat1-γ1*a, -bhat2+β2*a, chat2-γ2*a]
    A1 = [1 0 0 0 0; -1 0 0 0 0; β1 -1 0 0 0; -γ1 0 1 0 0; β2 0 0 -1 0; -γ2 0 0 0 1]
    b1 = zeros(T, 6)

    # Finally collect the outcomes
    A2 = [β1 -γ1 -1 1 0 0; β2 -γ2 0 0 -1 1]
    b2 = [exp(pivot); exp(-pivot)]
    M1 = ReLUNet([A1,A2],[b1,b2])

    hcat_model(M0, M1)
end
