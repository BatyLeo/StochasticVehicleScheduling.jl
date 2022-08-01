#using Gurobi
@info "Creating a GRB_ENV const for StochasticVehicleScheduling..."
# Gurobi package setup (see https://github.com/jump-dev/Gurobi.jl/issues/424)
const GRB_ENV = Ref{Gurobi.Env}()
GRB_ENV[] = Gurobi.Env()
export GRB_ENV
