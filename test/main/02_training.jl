using Random  # TODO: move seed into trainer/training loop
using StochasticVehicleScheduling

function main_imitation()
    files = [
        #"25tasks10scenarios.yaml",
        "50tasks50scenarios.yaml",
        "100tasks50scenarios.yaml",
        #"mixed.yaml",
        #"25tasks10scenarios_exact.yaml",
        # "25tasks10scenarios_garbage.yaml",
    ]
    for file in files
        Random.seed!(67);
        config_file = "test/main/configs/imitation/$file"
        trainer = Trainer(config_file);
        train_loop!(trainer)
    end
end

function main_experience()
    files = [
        #"mixed.yaml",
        #"25tasks10scenarios.yaml",
        "50tasks50scenarios.yaml",
        "100tasks50scenarios.yaml",
    ]
    for file in files
        Random.seed!(67);
        config_file = "test/main/configs/experience/$file"
        trainer = Trainer(config_file);
        train_loop!(trainer)
    end
end

# main_imitation()
main_experience()
