recursive_namedtuple(x::Any) = x
recursive_namedtuple(d::Dict) = namedtuple(Dict(k => recursive_namedtuple(v) for (k, v) in d))

# data = recursive_namedtuple(YAML.load_file("config.yaml"; dicttype=Dict{Symbol,Any}))

# (; name, config) = data.training.optimizer

# eval(Meta.parse(name))(config...)

function read_config(config_file::String)
    return recursive_namedtuple(YAML.load_file(config_file; dicttype=Dict{Symbol,Any}))
end
