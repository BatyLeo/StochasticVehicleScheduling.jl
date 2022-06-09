function test_perf(trainer::Trainer; test_name::String="Test")
    @testset "$test_name" begin
        for m_list in (trainer.metrics.train, trainer.metrics.test)
            for metric in m_list
                test_perf(metric)
            end
        end
    end
end

function plot_perf(trainer::Trainer; lineplot_function::Function)
    for m_list in (trainer.metrics.train, trainer.metrics.test)
        for metric in m_list
            plt = lineplot_function(metric.history; xlabel="Epoch", title=metric.name)
            println(plt)
        end
    end

    return nothing
end
