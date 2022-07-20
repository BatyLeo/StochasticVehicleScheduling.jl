using JLD2
using NamedTupleTools

results_dir = joinpath("final_experiments", "results")

imitation_model_mapping = Dict(
    "imitation_25tasks10scenarios" => "25 tasks",
    "imitation_50tasks50scenarios" => "50 tasks",
    "imitation_100tasks50scenarios" => "100 tasks",
    "imitation_mixed" => "Mixed",
)

experience_model_mapping = Dict(
    "experience_25tasks10scenarios" => "25 tasks",
    "experience_50tasks50scenarios" => "50 tasks",
    "experience_100tasks50scenarios" => "100 tasks",
    "experience_mixed" => "Mixed",
)

target_mapping = Dict(
    "25tasks10scenarios_exact_uncentered" => "25 tasks",
    "50tasks50scenarios_uncentered" => "50 tasks",
    "100tasks50scenarios_uncentered" => "100 tasks",
    "200tasks50scenarios_uncentered" => "200 tasks",
    "1000tasks10scenarios" => "1000 tasks",
)

files = readdir(results_dir)

function write_table(model_mapping, target_mapping; metric_name="average_cost_per_task", output_file="table.tex", caption="Figure")
    table = open(output_file, "w")
    # write header
    write(table, "\\begin{table}[H]\n")
    write(table, "\\centering\n")
    write(table, "\\begin{tabular}{$("|c"^(length(target_mapping)+1))|}\n\\hline\n")
    header = "Dataset"
    for column_name in values(target_mapping)
        header *= " & $column_name"
    end
    header *= "\\\\\n\\hline\n"
    write(table, header)
    # loop on rows
    for key in keys(model_mapping) # TODO: fix row order
        println(key)
        file = "$key.jld2"
        data = load(joinpath(results_dir, file))["data"]
        row = model_mapping[key]
        # loop on columns
        for target in keys(target_mapping)
            value = data[target][metric_name]
            row *= " & $(round(value, digits=3))"
        end
        row *= "\\\\\n\\hline\n"
        write(table, row)
    end
    # write ending
    write(table, "\\end{tabular}\n")
    write(table, "\\caption{$caption}\n")
    write(table, "\\end{table}\n")
    close(table)
end

metric = "average_cost_per_task"
#write_table(imitation_model_mapping, target_mapping; metric_name=metric)
write_table(experience_model_mapping, target_mapping; metric_name=metric)
