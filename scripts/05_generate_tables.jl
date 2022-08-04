using JLD2
#using NamedTupleTools

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

function gap_table(
    model_mapping, target_mapping; output_file="table.tex",
)
    table = open(output_file, "w")
    # write header
    # write(table, "\\begin{table}[H]\n")
    # write(table, "\\centering\n")
    nb_col = length(target_mapping) * 2
    write(table, "\\begin{tabular}{|c|$("c"^nb_col)|}\n\\hline\n")
    header = "\\multirow{3}{*}{\\textbf{Train dataset}} & \\multicolumn{$nb_col}{c|}{\\textbf{Test dataset}} \\\\ \\cline{2-$(nb_col+1)}\n"
    for (_, column_name) in values(target_mapping)
        header *= " & \\multicolumn{2}{c|}{$column_name}"
    end
    header *= "\\\\\n\\cline{2-$(nb_col+1)}\n"
    for (_, column_name) in values(target_mapping)
        header *= " & \\multicolumn{1}{c|}{mean} & \\multicolumn{1}{c|}{max}"
    end
    header *= "\\\\\n\\hline\n"
    write(table, header)
    # loop on rows
    for (key, row) in model_mapping
        println(key)
        file = "$key.jld2"
        data = load(joinpath(results_dir, file))["data"]
        # row = model_mapping[key]
        # loop on columns
        for (target, _) in target_mapping
            # value = data[target][metric_name]
            average_gap = data[target]["average_cost_gap"]
            max_gap = data[target]["max_cost_gap"]
            row *= " & \\multicolumn{1}{c|}{\$$(round(average_gap, digits=2))\\%\$} & \\multicolumn{1}{c|}{\$$(round(max_gap, digits=2))\\%\$}"
        end
        row *= "\\\\\n\\hline\n"
        write(table, row)
    end
    # write ending
    write(table, "\\end{tabular}\n")
    # write(table, "\\caption{$caption}\n")
    # write(table, "\\end{table}\n")
    return close(table)
end

function task_cost_table(
    model_mapping, target_mapping; output_file="table.tex"
)
    metric_name = "average_cost_per_task"
    table = open(output_file, "w")
    # write header
    # write(table, "\\begin{table}[H]\n")
    # write(table, "\\centering\n")
    nb_col = length(target_mapping)
    write(table, "\\begin{tabular}{|c|$("c"^nb_col)|}\n\\hline\n")
    header = "\\multirow{2}{*}{\\textbf{Train dataset}} & \\multicolumn{$nb_col}{c|}{\\textbf{Test dataset} (number of tasks in each instance)} \\\\ \\cline{2-$(nb_col+1)}\n"
    for (_, column_name) in values(target_mapping)
        header *= " & \\multicolumn{1}{c|}{$column_name}"
    end
    header *= "\\\\\n\\hline\n"
    write(table, header)
    # loop on rows
    for (key, row) in model_mapping
        println(key)
        file = "$key.jld2"
        data = load(joinpath(results_dir, file))["data"]
        # row = model_mapping[key]
        # loop on columns
        for (target, _) in target_mapping
            value = data[target][metric_name]
            row *= " & \\multicolumn{1}{c|}{$(round(value, digits=2))}"
        end
        row *= "\\\\\n\\hline\n"
        write(table, row)
    end
    # write ending
    write(table, "\\end{tabular}\n")
    # write(table, "\\caption{$caption}\n")
    # write(table, "\\end{table}\n")
    return close(table)
end

function generate_tables(
    imitation_model_mapping,
    experience_model_mapping,
    target_mapping_1,
    target_mapping_2
)
    table_dir = joinpath(figure_dir, "tables")

    gap_table(
        imitation_model_mapping, target_mapping_1;
        output_file=joinpath(table_dir, "imitation_gap.tex"),
    )
    task_cost_table(
        imitation_model_mapping, target_mapping_2;
        output_file=joinpath(table_dir, "imitation_cost.tex"),
    )

    gap_table(
        experience_model_mapping, target_mapping_1;
        output_file=joinpath(table_dir, "experience_gap.tex"),
    )
    task_cost_table(
        experience_model_mapping, target_mapping_2;
        output_file=joinpath(table_dir, "experience_cost.tex"),
    )
end

generate_tables(imitation_model_mapping, experience_model_mapping, target_mapping_1, target_mapping_2)
