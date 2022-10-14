```@meta
CurrentModule = StochasticVehicleScheduling
```

# StochasticVehicleScheduling

Documentation for [StochasticVehicleScheduling](https://github.com/BatyLeo/StochasticVehicleScheduling.jl). This package is a satellite of [InferOpt.jl](https://github.com/axelparmentier/InferOpt.jl). It illustrates Machine Learning/Combinatorial Optimization hybrid algorithms from `InferOpt.jl` applied to the Stochastic Vehicle Scheduling problem. It was used for one of the numerical experiments in this paper: [https://arxiv.org/abs/2207.13513](https://arxiv.org/abs/2207.13513)

If you have any feedback or question, feel free to [create an issue](https://github.com/BatyLeo/StochasticVehicleScheduling.jl/issues/new/choose) or [contact me](mailto:leo.baty@enpc.fr).

# Installation
```julia
using Pkg
Pkg.add(url="https://github.com/BatyLeo/StochasticVehicleScheduling.jl")
```

# Table of contents

- [Problem statement](@ref)
- [Instance generator](@ref)
- [Optimization algorithms](@ref)
- Learning using `InferOpt.jl`
  - [InferOpt tutorial](@ref)
  - [Reproducing paper experiments](@ref) from [https://arxiv.org/abs/2207.13513](https://arxiv.org/abs/2207.13513)
