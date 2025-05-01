
function compute_sorting_operations(L)
    N = 2^L
    steps = (L * (L+1)) >> 1
    sorts = Vector{Vector{Tuple{Int,Int,Int}}}(undef, steps)
    for step in 1:steps
        sorts[step] = Tuple{Int,Int,Int}[]
    end
    step = 0
    for i in 1:L
        for j in i-1:-1:0
            step += 1
            for k in 0:N-1
                l = xor(k,2^j) # bitwise XOR of k and 2^j
                if l > k
                    if (k & 2^i) == 0  # bitwise AND of k and 2^i
                        push!(sorts[step], (k+1, l+1, 1))  # sort increasing
                    else
                        push!(sorts[step], (k+1, l+1, 2))  # sort decreasing
                    end
                end
            end
        end
    end
    sorts
end

"Make a bitonic sorting network for vectors of size N = 2^L."
function bitonic_sort_network(L::Int, ::Type{T} = Float64) where {T}
    N = 2^L
    sorts = compute_sorting_operations(L)
    steps = length(sorts)

    sort2 = ann_sort2(T)
    sort_A1 = sort2.weights[1]
    sort_A2 = sort2.weights[2]

    models = Any[]
    for step in 1:steps
        nsorts = length(sorts[step])
        A1 = spzeros(T, 4*nsorts, N)
        A2 = spzeros(T, N, 4*nsorts)
        for s in 0:nsorts-1
            i,j,k = sorts[step][s+1]
            if k == 2       # we always sort increasing and switch i and j otherwise
                j,i = i,j
            end
            A1[4*s+1:4*s+4,i] = sort_A1[:,1]
            A1[4*s+1:4*s+4,j] = sort_A1[:,2]
            A2[i,4*s+1:4*s+4] = sort_A2[1,:]
            A2[j,4*s+1:4*s+4] = sort_A2[2,:]
        end
        push!(models, ReLUNet([A1,A2]))
    end
    reduce(hcat_model, models)
end
