@info "Hello"
using Gurobi
# Gurobi package setup (see https://github.com/jump-dev/Gurobi.jl/issues/424)
const GRB_ENV = Ref{Gurobi.Env}()
GRB_ENV[] = Gurobi.Env()
