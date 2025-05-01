function mul_approx_refine_x(x, y, j = 1; L = 20, init = :zero)
    T = typeof(x)
    if j == L+1
        if init == :zero
            return zero(T)
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    z = mul_approx_refine_x(pivot - abs(x-pivot), y, j+1; L, init)
    if x > pivot
        2*pivot*y - z
    else
        z
    end
end

function mul_approx(x, y, j = 1; L = 20, init = :zero)
    T = typeof(x)
    if j == L+1
        if init == :zero
            return zero(T)
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    z = mul_approx_y(pivot - abs(x-pivot), y, j; L, init)
    if x > pivot
        2*pivot*y - z
    else
        z
    end
end

function mul_approx_y(x, y, j; L, init)
    T = typeof(x)
    pivot = one(T)/2^j
    z = mul_approx(x, pivot - abs(y-pivot), j+1; L, init)
    if y > pivot
        2*pivot*x - z
    else
        z
    end
end

function mul_approx_relu(x, y, j = 1; L = 20, init = :zero)
    T = typeof(x)
    if j == L+1
        if init == :zero
            return zero(T)
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = one(T)/2^j
    hx = pivot - relu(x-pivot) - relu(pivot-x)
    z = mul_approx_y_relu(hx, y, j; L, init)

    a = x - pivot
    # Following the general methodology we obtain:
    # bhat = pivot*y - z
    # chat = z - pivot*y
    # u = pivot*y + relu(a) - relu(-a) - relu(-bhat+a) + relu(chat-a)
    # However, since bhat and chat agree up to a sign, we need only one.
    # (This is a sign that we could simplify more here.)
    bhat = pivot*y - z
    u = pivot*y + relu(a) - relu(-a) - relu(-bhat+a) + relu(-bhat-a)
    u
end

function mul_approx_y_relu(x, y, j; L, init)
    T = typeof(x)
    pivot = one(T)/2^j
    hy = pivot - relu(y-pivot) - relu(pivot-y)
    z = mul_approx_relu(x, hy, j+1; L, init)

    a = y - pivot
    bhat = pivot*x - z

    u = pivot*x + relu(a) - relu(-a) - relu(-bhat+a) + relu(-bhat-a)
    u
end


function ann_mul(L, ::Type{T} = Float64; init = :zero) where {T}
    Mperm = ann_permute2(T)
    Mref = ann_refine_2d(L, T)
    M0 = hcat_model(Mperm, Mref)

    @assert init == :zero
    A0 = zeros(T, 2L+2, 2L+2)
    A0[2:end,2:end] = I(2L+1)
    Minit = ReLUNet([A0])

    M = hcat_model(M0, Minit)

    for j in L:-1:1
        Mj = ann_mul_rec(j, T; Mr=M)
        M = hcat_model_skipremainder(M, ann_mul_rec(j, T; Mr = M))
    end
    # From the last output [z; x], select z
    Mselect = ann_select(1, 2, T)
    hcat_model(M, Mselect)
end

"Map [z_{j+1}, x_{j+1}, y_j, x_j] to [z_j, x_j]."
function ann_mul_rec(j, ::Type{T} = Float64; Mr) where {T}
    pivot = one(T)/2^j
    # First we unfold y.
    # 1) map from [z_{j+1}, x_{j+1}, y_j, x_j] to [a, bhat, x_{j+1}, y_j, x_j]
    #    with a = y_j - pivot and bhat = pivot*x_{j+1}-z_{j+1}
    A0 = [0 0 1 0; -1 pivot 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    b0 = [-pivot; 0; 0; 0; 0]
    M0 = ReLUNet([A0],[b0])

    # 2) Form the inputs: a, -a, -bhat+a, -bhat-a, x_{j+1}, -x_{j+1}, y_j, -y_j, x_j, -x_j
    A1 = zeros(T, 10, 5)
    A1 .= [1 0 0 0 0; -1 0 0 0 0; 1 -1 0 0 0; -1 -1 0 0 0;
        0 0 1 0 0; 0 0 -1 0 0; 0 0 0 1 0; 0 0 0 -1 0; 0 0 0 0 1; 0 0 0 0 -1]
    # 3) Form the outputs: [z, y_j, x_j]
    A2 = zeros(T, 3, 10)
    # Recall: z = pivot*x_{j+1} + relu(a) - relu(-a) - relu(-bhat+a) + relu(-bhat-a)
    A2 .= [1 -1 -1 1 pivot -pivot 0 0 0 0;
        0 0 0 0 0 0 1 -1 0 0;
        0 0 0 0 0 0 0 0 1 -1]
    M1 = ReLUNet([A1,A2])

    My = hcat_model(M0, M1)

    # Then we unfold x.
    # 1) map from [z, y, x] to [a, bhat, y, x]
    #    with a = x-pivot and bhat = pivot*y-z
    A0x = T[0 0 1; -1 pivot 0; 0 1 0; 0 0 1]
    b0x = [-pivot; 0; 0; 0]
    M0x = ReLUNet([A0x], [b0x])
    # 2) Form the inputs: a, -a, -bhat+a, -bhat-a, y, -y, x, -x
    A1x = T[1 0 0 0; -1 0 0 0; 1 -1 0 0; -1 -1 0 0;
        0 0 1 0; 0 0 -1 0; 0 0 0 1; 0 0 0 -1]
    # 3) Form the output: [z,x_j]
    # Recall: z = pivot*y + relu(a) - relu(-a) - relu(-bhat+a) + relu(-bhat-a)
    A2x = T[1 -1 -1 1 pivot -pivot 0 0;
        0 0 0 0 0 0 1 -1]
    M1x = ReLUNet([A1x,A2x])
    Mx = hcat_model(M0x, M1x)
    M = hcat_model(My, Mx)
end
