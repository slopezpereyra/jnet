

X = MNIST(:train)
NETWORK = Network([784, 16, 16, 10])
NETWORK.net

network = f_propagation(X[10].features, NETWORK).net
avg_cost = train(X, NETWORK)