function cbc_model()
    model = Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "logLevel", 0)
    return model
end

function glpk_model()
    model = Model(GLPK.Optimizer)
    return model
end
