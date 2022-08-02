@info "Creating a GRB_ENV const for StochasticVehicleScheduling..."
# Gurobi package setup (see https://github.com/jump-dev/Gurobi.jl/issues/424)
const GRB_ENV = Ref{Gurobi.Env}()
GRB_ENV[] = Gurobi.Env()
export GRB_ENV

function grb_model()
    model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    set_optimizer_attribute(model, "OutputFlag", 0)
    return model
end

export grb_model
