using Random
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training
using UnicodePlots

Random.seed!(67);
config_file = "src/training/config.yaml"
trainer = Trainer(config_file);
train_loop!(trainer)

# ---

#plot_perf(trainer; lineplot_function=lineplot)

using TensorBoardLogger, ValueHistories, GLMakie

# r = TBReader("logs/learn_by_imitation")
# r = TBReader("logs/learning_by_experience")
r = TBReader("logs/test_new_inferopt_normalized_5")

hist = MVHistory()

TensorBoardLogger.map_summaries(r) do tag, iter, val
    push!(hist, Symbol(tag), iter, val)
end

hist

tags = Dict(
    "train/loss" => "Train loss",
    "train/max_cost_gap" => "Train max cost gap",
    "train/average_cost_gap" => "Train average cost gap",
    "test/loss" => "Test loss",
    "test/max_cost_gap" => "Test max cost gap",
    "test/average_cost_gap" => "Test average cost gap",
    #"test/average_perturbed_cost_gap" => "Test average cost gap",
)

for (tag, value) in tags
    y = hist[tag].values
    x = hist[tag].iterations
    #println(lineplot(x, y; title=tag))
    #lines(x, y)
    fig = Figure()
    if occursin("cost_gap", tag)
        ax = Axis(fig[1, 1], yscale=log10)
        @info "y" y
        for i in eachindex(y)
            y[i] = max(y[i], 1e-5)
        end
        # y .= max.(y, 1e-10)
        # println("hello")
        @info "xy" x y
    else
        ax = Axis(fig[1, 1])
    end
    ax.xlabel = "epochs"

    ax.title = value
    slice = 1:length(x)
    GLMakie.lines!(ax, x[slice], y[slice])
    save("figures/normalized/$(replace(tag, "/"=>"_")).png", fig)
end
