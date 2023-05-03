# This file implements Metric type, which allow automatized metric computation

abstract type AbstractMetric end

function compute_value!(m::AbstractMetric, t::Trainer; kwargs...)
    return push!(m.history, m(t; kwargs...))
end

function test_perf(metric::AbstractMetric)
    @test metric.history[end] < metric.history[1]
end

function log_last_measure!(m::AbstractMetric, logger=nothing; train=true, step_increment=0) end

## ---

mutable struct Loss <: AbstractMetric
    name::String
    history::Vector{Float64}
    best_value::Float64
end

Loss(name="Loss") = Loss(name, Float64[], Inf)

function (m::Loss)(trainer::Trainer; train, epoch, kwargs...)
    data = train ? trainer.data.train : trainer.data.validation
    value = mean(trainer.loss(t...) for t in loader(data))
    return value
end

function log_last_measure!(m::Loss, logger::AbstractLogger; train=true, step_increment=0)
    str = train ? "train" : "validation"
    with_logger(logger) do
        @info "$str" loss = m.history[end] log_step_increment = step_increment
    end
end

## ---

mutable struct AllCosts <: AbstractMetric
    name::String
    best_value::Float64
    average_cost_gap::Vector{Float64}
    max_cost_gap::Vector{Float64}
    average_cost_per_task::Vector{Float64}
end

AllCosts(name="All costs") = AllCosts(name, Inf, Float64[], Float64[], Float64[])

function compute_value!(m::AllCosts, t::Trainer; kwargs...)
    a, b, c = m(t; kwargs...)
    push!(m.average_cost_gap, a)
    push!(m.max_cost_gap, b)
    return push!(m.average_cost_per_task, c)
end

function (m::AllCosts)(trainer::Trainer; train, epoch, Y_pred, kwargs...)
    (; cost) = trainer
    data = train ? trainer.data.train : trainer.data.validation
    train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    average_cost_gap =
        mean((c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)) *
        100
    max_cost_gap =
        maximum(
            (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
        ) * 100
    average_cost_per_task = mean(
        c / x.city.nb_tasks for (c, c_opt, x) in zip(train_cost, train_cost_opt, data.X)
    )

    if !train && average_cost_per_task < m.best_value
        m.best_value = average_cost_per_task
        save_model(trainer, epoch; best=true)
    end
    return average_cost_gap, max_cost_gap, average_cost_per_task#, max_cost
end

function log_last_measure!(
    m::AllCosts, logger::AbstractLogger; train=true, step_increment=0
)
    str = train ? "train" : "validation"
    with_logger(logger) do
        @info "$str" average_cost_gap = m.average_cost_gap[end] log_step_increment =
            step_increment
        @info "$str" max_cost_gap = m.max_cost_gap[end] log_step_increment = 0
        @info "$str" average_cost_per_task = m.average_cost_per_task[end] log_step_increment =
            0
    end
end
