using GLMakie
using TensorBoardLogger
using ValueHistories

const logdir = "logs"
const figure_dir = "figures/plots"

const imitation_models = [
    ("imitation_25tasks10scenarios", "25 tasks"),
    ("imitation_50tasks50scenarios", "50 tasks"),
    ("imitation_100tasks50scenarios", "100 tasks"),
]

const experience_models = [
    ("experience_25tasks10scenarios", "25 tasks"),
    ("experience_50tasks50scenarios", "50 tasks"),
    ("experience_100tasks50scenarios", "100 tasks"),
]

const tags = [
    ("loss", "Loss"),
    ("average_cost_gap", "Average cost gap in %"),
    ("max_cost_gap", "Max cost gap in %"),
    ("average_cost_per_task", "Average cost per task"),
]

function read_model(model_dir)
    r = TBReader(model_dir)
    hist = MVHistory()
    TensorBoardLogger.map_summaries(r) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    return hist
end

function cool_plot(models, tags)
    f = Figure(; backgroundcolor=RGBf(1.0, 1.0, 1.0), resolution=(800, 1200))
    ga = f[1, 1] = GridLayout()

    local ax
    for (irow, (tag, value)) in enumerate(tags)
        ax_list = []
        for (icol, (model, text)) in enumerate(models)
            model_dir = joinpath(logdir, model)
            hist = read_model(model_dir)

            ax = Axis(ga[irow + 1, icol]; xlabel="epochs")
            # Link axes with previous plot if there is one
            if icol > 1
                linkxaxes!(ax_list[icol - 1], ax)
            end
            train_tag = "train/$tag"
            validation_tag = "validation/$tag"
            xtrain = hist[train_tag].iterations
            ytrain = hist[train_tag].values
            xvalidation = hist[validation_tag].iterations
            yvalidation = hist[validation_tag].values

            lines!(ax, xtrain, ytrain; label="Training metric")
            lines!(ax, xvalidation, yvalidation; label="Validation metric")
            ax.xticks = 0:(xtrain[end] / 5):xtrain[end]

            push!(ax_list, ax)

            Label(ga[irow + 1, icol, Top()], "$text"; valign=:bottom, padding=(0, 0, 1, 0))
        end
        Label(
            ga[irow + 1, 1:length(ax_list), Top()],
            "$value";
            valign=:bottom,
            font="TeX Gyre Heros Bold",
            padding=(0, 0, 20, 0),
        )
    end

    leg = Legend(ga[1, 2], ax)
    leg.tellwidth = false
    leg.tellheight = true

    colgap!(ga, 10)
    rowgap!(ga, 5)
    return f
end

fig1 = cool_plot(imitation_models, tags);
fig2 = cool_plot(experience_models, tags);

save("figures/plots/imitation.png", fig1)
save("figures/plots/experience.png", fig2)
