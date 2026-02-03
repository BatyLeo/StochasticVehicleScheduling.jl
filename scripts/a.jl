using Random  # TODO: move seed into trainer/training loop
using StochasticVehicleScheduling
using Flux
using Gurobi

Random.seed!(67)
file = "25tasks10scenarios.yaml"
config_file = "scripts/configs/experience/$file"
trainer = Trainer(config_file);

(; nb_epochs, loss, pipeline, opt, data) = trainer;

pipeline_loss(X) = mean(loss(encoder(x.features); instance=x) for x in X)

for n in 1:nb_epochs
    ps = Flux.params(pipeline.encoder)
    for batch in data
        # gs = Flux.gradient(ps) do
        #     return loss(batch...)
        # end
        println(eltype(collect(batch)))
        println(typeof(batch...))
        println(pipeline_loss(batch...))
        # Flux.update!(opt, ps, gs)
    end
end
