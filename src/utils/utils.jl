struct Point
    x::Float64
    y::Float64
end

function draw_random_point(law::Distribution)
    return Point(rand(law), rand(law))
end

function distance(p₁::Point, p₂::Point)
    return sqrt((p₁.x - p₂.x) * (p₁.x - p₂.x) + (p₁.y - p₂.y) * (p₁.y - p₂.y))
end

"""
    hour_of(minutes::Real)::Int

Returns hour of the day corresponding to minutes amount
"""
function hour_of(minutes::Real)::Int
    return min(24, trunc(Int, minutes / 60) + 1)
end

function find_first_one(A::AbstractVector)
    for (i, elem) in enumerate(A)
        if elem
            return i
        end
    end
    return nothing
end
