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

# Config stuff
"""
    recursive_namedtuple(x)

Convert recursively a Dict to a NamedTuple.
"""
recursive_namedtuple(x::Any) = x
function recursive_namedtuple(d::Dict)
    return namedtuple(Dict(k => recursive_namedtuple(v) for (k, v) in d))
end

"""
    recursive_convert(x)

Convert recursively a NamedTuple to a Dict.
"""
recursive_convert(x::Any) = x
function recursive_convert(x::NamedTuple)
    nt = NamedTuple((k, recursive_convert(v)) for (k, v) in zip(keys(x), x))
    return convert(Dict, nt)
end

"""
    read_config(config_file::String)

Read a Yaml config into a NamedTuple.
"""
function read_config(config_file::String)
    return recursive_namedtuple(YAML.load_file(config_file; dicttype=Dict{Symbol,Any}))
end

"""
    save_config(config::NamedTuple, save_path::String)

Save a NamedTuple config to yaml file.
"""
function save_config(config::NamedTuple, save_path::String)
    return YAML.write_file(save_path, recursive_convert(config))
end

"""
    save_config(config::Dict, save_path::String)

Save Dict config to yaml file.
"""
function save_config(config::Dict, save_path::String)
    return YAML.write_file(save_path, config)
end
