using Random
using UnicodePlots

function short(solution::Solution)
    res = Vector{Int}[]
    n, m = size(solution.path_value)
    for row in 1:n
        hello = Int[]
        for col in 1:m
            if solution.path_value[row, col]
                push!(hello, col)
            end
        end
        if length(hello) > 0
            push!(res, hello)
        end
    end
    return "$(length(res)), $res"
end
