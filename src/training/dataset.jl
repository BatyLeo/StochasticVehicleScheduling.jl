# struct InferOptDataset{D}
#     train::D
#     test::D
# end

# function InferOptDataset(;
#     X_train=nothing,
#     thetas_train=nothing,
#     Y_train=nothing,
#     X_test=nothing,
#     thetas_test=nothing,
#     Y_test=nothing,
# )
#     data_train = (X=X_train, thetas=thetas_train, Y=Y_train)
#     data_test = (X=X_test, thetas=thetas_test, Y=Y_test)
#     return InferOptDataset(data_train, data_test)
# end

# function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
#     N = length(X)
#     N_train = floor(Int, N * train_percentage)
#     N_test = N - N_train
#     train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
#     X_train, X_test = X[train_ind], X[test_ind]
#     return X_train, X_test
# end

abstract type AbstractDataset end

struct SupervisedDataset{Dx, Dy} <: AbstractDataset
    X::Dx
    Y::Dy
end

function loss_data(dataset::SupervisedDataset)
    return zip(dataset.X, dataset.Y)
end

struct ExperienceDataset{Dx, Dy} <: AbstractDataset
    X::Dx
    Y::Dy
end

function loss_data(dataset::ExperienceDataset)
    return zip(dataset.X)
end
