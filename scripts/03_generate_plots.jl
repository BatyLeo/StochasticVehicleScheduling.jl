using GLMakie
using TensorBoardLogger
using ValueHistories

const logdir = "logs"
const figure_dir = "figures/plots"
const Δ = 10.0

const models = Dict(
    "imitation_25tasks10scenarios" => "25 tasks",
    "imitation_50tasks50scenarios" => "50 tasks",
    "imitation_100tasks50scenarios" => "100 tasks",
    "experience_25tasks10scenarios" => "25 tasks",
    "experience_50tasks50scenarios" => "50 tasks",
    "experience_100tasks50scenarios" => "100 tasks",
)

const tags = Dict(
    "loss" => "Loss",
    "max_cost_gap" => "Max cost gap",
    "average_cost_gap" => "Average cost gap",
    "average_cost_per_task" => "Average cost per task",
)

function read_model(model_dir)
    r = TBReader(model_dir)
    hist = MVHistory()
    TensorBoardLogger.map_summaries(r) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    return hist
end

function generate_plots(models, tags)
    for (model, text) in models
        model_dir = joinpath(logdir, model)
        hist = read_model(model_dir)

        for (tag, value) in tags
            train_tag = "train/$tag"
            validation_tag = "validation/$tag"
            xtrain = hist[train_tag].iterations
            ytrain = hist[train_tag].values
            xvalidation = hist[validation_tag].iterations
            yvalidation = hist[validation_tag].values
            if occursin("cost gap", value)
                ytrain .+= Δ
                yvalidation .+= Δ
            end
            # for i in eachindex(ytrain)
            #     ytrain[i] = max(ytrain[i], 1e-5)
            # end
            # for i in eachindex(yvalidation)
            #     yvalidation[i] = max(yvalidation[i], 1e-5)
            # end

            fig = Figure()
            ax = Axis(fig[1, 1]; yscale=log10)
            ax.xlabel = "epochs"

            ax.title = value
            ltrain = GLMakie.lines!(ax, xtrain, ytrain)
            lvalidation = GLMakie.lines!(ax, xvalidation, yvalidation)
            Legend(fig[1, 2], [ltrain, lvalidation], ["train", "validation"])
            save(joinpath(figure_dir, "$(model)_$(replace(tag, "/"=>"_")).png"), fig)
        end
    end
end

generate_plots(models, tags)