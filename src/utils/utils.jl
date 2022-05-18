struct Point
    x::Float64
    y::Float64
end

"""
    draw_random_point(distrib)

Returns a Point with random x and y, drawn from distrib.
"""
function draw_random_point(distrib::Distribution)
    return Point(rand(distrib), rand(distrib))
end

"""
    distance(p₁, p₂)

Returns euclidean distance between p₁ and p₂.
"""
function distance(p₁::Point, p₂::Point)
    return sqrt((p₁.x - p₂.x) * (p₁.x - p₂.x) + (p₁.y - p₂.y) * (p₁.y - p₂.y))
end

"""
    hour_of(minutes::Real)::Int

Returns hour of the day corresponding to minutes amount.
"""
function hour_of(minutes::Real)::Int
    return min(24, trunc(Int, minutes / 60) + 1)
end

"""
    find_first_one(A)

Returns index of first non zero element of A.
"""
function find_first_one(A::AbstractVector)
    for (i, elem) in enumerate(A)
        if elem
            return i
        end
    end
    return nothing
end
