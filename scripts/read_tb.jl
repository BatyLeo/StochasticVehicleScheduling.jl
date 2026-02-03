using TensorBoardLogger
using ValueHistories

function read_model(model_dir)
    r = TBReader(model_dir)
    hist = MVHistory()
    TensorBoardLogger.map_summaries(r) do tag, iter, val
        push!(hist, Symbol(tag), iter, val)
    end
    return hist
end

read_model(joinpath("logs", "experience_25tasks10scenarios"))

model_dir = joinpath("logs", "test", "a")
model_dir = joinpath("logs_old", "hyperopt_variance_reduction", "1.0_44.0")
r = TBReader(model_dir)
hist = MVHistory()

TensorBoardLogger.map_summaries(r) do tag, iter, val
    push!(hist, Symbol(tag), iter, val)
end
