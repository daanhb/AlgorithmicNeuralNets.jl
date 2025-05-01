function square_approx(x::Number, j = 1; L = 20, init=:taylor)
    T = typeof(x)
    if j == L+1
        if init == :taylor
            return zero(T)
        elseif init == :midvalue
            return one(T)/2^(2L+1)
        elseif init == :interpolation
            return x/2^L
        else
            throw(ArgumentError("Unknown initialization: :$(init)"))
        end
    end
    pivot = 1/2^j
    z = square_approx(pivot - abs(x-pivot), j+1; L, init)
    if x > pivot
        return z + 4*pivot*x - 4*pivot^2
    else
        return z
    end
end


"""
    ann_square(levels[, T]; options...)

Return a neural network to approximate the square function.

Optionall, the keyword argument init can be supplied to select among different
types of approximation. Possible values are the symbols :taylor, :interpolation and
:midvalue.
"""
function ann_square(levels::Int, ::Type{T} = Float64; init=:taylor) where {T}
    models = [ann_square_rec(level, T) for level in 1:levels]
    M = reduce(hcat_model, models)
    # Make input go from [x] to [x; 0]
    Min = ann_input(1, 2, T)

    # From [x; z] compute the final z value.
    if init == :taylor
        # from [x; z] select z
        Mselect = ann_select(2, 2, T)
    elseif init == :midvalue
        Mselect = ann_select(2, 2, T)
        Mselect.biases[1][1] = one(T)/2^(2*levels+1)
    elseif init == :interpolation
        Mselect = ann_select(2, 2, T)
        Mselect.weights[1][1] = one(T)/2^levels
    else
        throw(ArgumentError("Unknown initialization: :$(init)"))
    end
    hcat_model(Min, M, Mselect)
end

"Map [x,z] to [s - abs(x-s); z -2*s^2 + 2*x*s + 2*s*abs(x-s)] where s is the pivot at the given level."
function ann_square_rec(level, ::Type{T} = Float64) where T
    s = one(T) / 2^level
    A1 = T[1 0; -1 0; 2s 1; -2s -1]
    b1 = [-s; s; -2s^2; 2s^2]
    A2 = T[-1 -1 0 0; 2s 2s 1 -1]
    b2 = [s; 0]
    M1 = ReLUNet([A1,A2],[b1,b2])
end


"Map [x,z] to [abs(x-2^(-l)),z+1/2^(l-1)*x - 1/2^(2l)]."
function sqr_refine_rec_alternative(l, ::Type{T} = Float64) where T
    s0 = one(T)/2^(l-1)
    s1 = one(T)/2^l
    s2 = one(T)/2^(2l)
    # the first map is simply affine
    A1 = [1 0; s0 1]
    b1 = [-s1; -s2]
    M1 = ReLUNet([A1],[b1])
    # the second map is abs for the first input and identity for the second
    M2 = vcat_model(ann_abs(T), ann_id(T))
    hcat_model(M1, M2)
end


function ann_square_alternative(levels::Int, ::Type{T} = Float64) where {T}
    models = [sqr_refine_rec_alternative(level, T) for level in 1:levels]
    M = reduce(hcat_model, models)
    Min = ann_input(1, 2, T)
    Mselect = ann_select(2, 2, T)
    hcat_model(Min, M, Mselect)
end

