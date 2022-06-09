recursive_namedtuple(x::Any) = x
recursive_namedtuple(d::Dict) = namedtuple(Dict(k => recursive_namedtuple(v) for (k, v) in d))

recursive_convert(x::Any) = x
function recursive_convert(x::NamedTuple)
    nt = NamedTuple((k, recursive_convert(v)) for (k, v) in zip(keys(x), x))
    return convert(Dict, nt)
end

function read_config(config_file::String)
    return recursive_namedtuple(YAML.load_file(config_file; dicttype=Dict{Symbol,Any}))
end

function save_config(config::NamedTuple, save_path::String)
    YAML.write_file(save_path, recursive_convert(config))
end
