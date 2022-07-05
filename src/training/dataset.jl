# This file implements Dataset types that allow to use Flux loader and custom batchsize values

abstract type AbstractDataset end

loader(d::AbstractDataset) = d.loader

struct SupervisedDataset{Dx, Dy, L} <: AbstractDataset
    X::Dx
    Y::Dy
    loader::L
end

function SupervisedDataset(X, Y)
    return SupervisedDataset(X, Y, Flux.DataLoader((X, Y); batchsize=1))
end

function loss_data(dataset::SupervisedDataset)
    return zip(dataset.X, dataset.Y)
end

function build_loader(dataset::SupervisedDataset, batchsize)
    return Flux.DataLoader((dataset.X, dataset.Y); batchsize=batchsize)
end

struct ExperienceDataset{Dx, Dy, L} <: AbstractDataset
    X::Dx
    Y::Dy
    loader::L
end

function ExperienceDataset(X, Y)
    return ExperienceDataset(X, Y, Flux.DataLoader((X, ); batchsize=1))
end

function loss_data(dataset::ExperienceDataset)
    return zip(dataset.X)
end

function build_loader(dataset::ExperienceDataset, batchsize)
    return Flux.DataLoader((dataset.X, ); batchsize=batchsize)
end
