"""
    cbc_model()

Initialiaze a Cbc model with disabled logging.
"""
function cbc_model()
    model = Model(Cbc.Optimizer)
    set_optimizer_attribute(model, "logLevel", 0)
    return model
end

"""
    glpk_model()

Initialize a GLPK model (with disabled logging).
"""
function glpk_model()
    model = Model(GLPK.Optimizer)
    return model
end
