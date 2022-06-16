using Flux
using GeometricFlux
using Graphs
using Random
using StochasticVehicleScheduling
using StochasticVehicleScheduling.Training

Random.seed!(67);
config_file = "src/training/config.yaml"
trainer = Trainer(config_file);

x = trainer.data.train.X[1];
g = x.graph
nb_vertices = nv(g)

nb_features = 5
#fg = FeaturedGraph(g; nf=randn(nb_features, nb_vertices))
fg = FeaturedGraph(g; nf=randn(nb_features, nb_vertices), ef=x.features)
nv(fg), ne(fg)
edge_feature(fg)
node_feature(fg)

gc = GCNConv(nb_features=>20)

fg2 = gc(fg)
node_feature(fg2)

##
# g = [[2,3], [1,4,5], [1], [2,5], [2,4]]
# graph(FeaturedGraph(g))

# g = path_graph(10)
# fg = FeaturedGraph(g, nf=randn(nb_features, nb_vertices))
# gc = GCNConv(nb_features=>20)
# gc(fg)
# nv(fg), ne(fg)
# edge_feature(fg)
# node_feature(fg)
gc2 = GraphParallel(node_layer=Dense(20, nb_vertices))

res = gc2(gc(fg))
node_feature(res)

function build_θ(fg)
    features = node_feature(fg)
    return [features[e[1], e[2]] for (i, e) in edges(fg)]
end

build_θ(res)

GNN = Chain(
    GCNConv(nb_features=>20),
    GraphParallel(node_layer=Dense(20, nb_vertices)),
    build_θ
)
