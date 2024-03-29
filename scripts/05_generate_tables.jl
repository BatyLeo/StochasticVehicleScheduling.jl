using JLD2

log_dir = "logs"
figure_dir = "figures"
results_dir = joinpath(log_dir, "results")

imitation_model_mapping = [
    ("imitation_25tasks10scenarios", "25 tasks"),
    ("imitation_50tasks50scenarios", "50 tasks"),
    ("imitation_100tasks50scenarios", "100 tasks"),
]

experience_model_mapping = [
    ("experience_25tasks10scenarios", "25 tasks"),
    ("experience_50tasks50scenarios", "50 tasks"),
    ("experience_100tasks50scenarios", "100 tasks"),
]

target_mapping_1 = [
    ("25tasks10scenarios", "25 tasks"),
    ("50tasks50scenarios", "50 tasks"),
    ("100tasks50scenarios", "100 tasks"),
]

target_mapping_2 = [
    ("25tasks10scenarios", "25"),
    ("50tasks50scenarios", "50"),
    ("100tasks50scenarios", "100"),
    ("200tasks50scenarios", "200"),
    ("300tasks10scenarios", "300"),
    ("500tasks10scenarios", "500"),
    ("750tasks10scenarios", "750"),
    ("1000tasks10scenarios", "1000"),
]

files = readdir(results_dir)

function gap_table(model_mapping, target_mapping; output_file="table.tex")
    table = open(output_file, "w")
    # write header
    nb_col = length(target_mapping) * 2
    write(table, "\\begin{tabular}{c$("c"^nb_col)}\n\\toprule\n")
    header = "\\multirow{3}{*}{\\textbf{Train dataset}} & \\multicolumn{$nb_col}{c}{\\textbf{Test dataset}} \\\\ \\cline{2-$(nb_col+1)}\n"
    for (_, column_name) in values(target_mapping)
        header *= " & \\multicolumn{2}{c}{$column_name}"
    end
    header *= "\\\\\n\\cline{2-$(nb_col+1)}\n"
    for (_, column_name) in values(target_mapping)
        header *= " & mean & max"
    end
    write(table, header)
    # loop on rows
    for (key, row) in model_mapping
        println(key)
        new_row = "\\\\\n\\midrule\n" * row
        file = "$key.jld2"
        data = load(joinpath(results_dir, file))["data"]
        # loop on columns
        for (target, _) in target_mapping
            average_gap = data[target]["average_cost_gap"]
            max_gap = data[target]["max_cost_gap"]
            new_row *= " & \$$(round(average_gap, digits=2))\\%\$ & \$$(round(max_gap, digits=2))\\%\$"
        end
        write(table, new_row)
    end
    # write ending
    write(table, "\\\\\n\\bottomrule\n\\end{tabular}\n")
    return close(table)
end

function task_cost_table(model_mapping, target_mapping; output_file="table.tex")
    metric_name = "average_cost_per_task"
    table = open(output_file, "w")
    # write header
    nb_col = length(target_mapping)
    write(table, "\\begin{tabular}{c$("c"^nb_col)}\n\\toprule\n")
    header = "\\multirow{2}{*}{\\textbf{Train dataset}} & \\multicolumn{$nb_col}{c}{\\textbf{Test dataset} (number of tasks in each instance)} \\\\ \\cline{2-$(nb_col+1)}\n"
    for (_, column_name) in values(target_mapping)
        header *= " & $column_name"
    end
    write(table, header)
    # loop on rows
    for (key, row) in model_mapping
        println(key)
        new_row = "\\\\\n\\midrule\n" * row
        file = "$key.jld2"
        data = load(joinpath(results_dir, file))["data"]
        # loop on columns
        for (target, _) in target_mapping
            value = data[target][metric_name]
            new_row *= " & $(round(value, digits=2))"
        end
        write(table, new_row)
    end
    # write ending
    write(table, "\\\\\n\\bottomrule\n\\end{tabular}\n")
    return close(table)
end

function time_table(model_mapping, target_mapping; output_file="table.tex")
    table = open(output_file, "w")
    # write header
    nb_col = length(target_mapping) * 2
    write(table, "\\begin{tabular}{$("c"^nb_col)}\n\\toprule\n")
    header = "\\multicolumn{$nb_col}{c}{\\textbf{Test dataset}} \\\\ \\midrule\n"
    for (_, column_name) in values(target_mapping)
        header *= "\\multicolumn{2}{c}{$column_name} & "
    end
    header = header[1:(end - 3)]
    header *= "\\\\\n\\midrule\n"
    for (_, column_name) in values(target_mapping)
        header *= "heuristic & learned pipeline & "
    end
    write(table, header[1:(end - 3)])
    # loop on rows
    for (key, row) in model_mapping
        println(key)
        new_row = "\\\\\n\\midrule\n"# * row
        file = "$key.jld2"
        data = load(joinpath(results_dir, file))["data"]
        # loop on columns
        for (target, _) in target_mapping
            elapsed_time = data[target]["elapsed_time"]
            elapsed_time_heuristic = data[target]["elapsed_time_heuristic"]
            new_row *= "\$$(round(elapsed_time_heuristic, digits=2))s\$ & \$$(round(elapsed_time, digits=3))s\$ & "
        end
        write(table, new_row[1:(end - 3)])
    end
    # write ending
    write(table, "\\\\\n\\bottomrule\n\\end{tabular}\n")
    return close(table)
end

function generate_tables(
    imitation_model_mapping, experience_model_mapping, target_mapping_1, target_mapping_2
)
    table_dir = joinpath(figure_dir, "tables")

    gap_table(
        imitation_model_mapping,
        target_mapping_1;
        output_file=joinpath(table_dir, "imitation_gap.tex"),
    )
    task_cost_table(
        imitation_model_mapping,
        target_mapping_2;
        output_file=joinpath(table_dir, "imitation_cost.tex"),
    )

    gap_table(
        experience_model_mapping,
        target_mapping_1;
        output_file=joinpath(table_dir, "experience_gap.tex"),
    )
    task_cost_table(
        experience_model_mapping,
        target_mapping_2;
        output_file=joinpath(table_dir, "experience_cost.tex"),
    )

    time_table(
        imitation_model_mapping[1:1],
        target_mapping_1;
        output_file=joinpath(table_dir, "imitation_time.tex"),
    )
    return time_table(
        experience_model_mapping[1:1],
        target_mapping_1;
        output_file=joinpath(table_dir, "experience_time.tex"),
    )
end

generate_tables(
    imitation_model_mapping, experience_model_mapping, target_mapping_1, target_mapping_2
)
