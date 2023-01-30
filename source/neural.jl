nclude("math.jl")
include("neural_structure.jl")
using LinearAlgebra
using MLDatasets: MNIST
using Random
using Statistics
using Serialization
using Shuffle

function compute_layer(W::Matrix, a::Vector, b::Vector)
    Y = zeros(Float32, length(b))
    mul!(Y, W, a)
    Y = Y + b
    return Y
end

function fprop(x::Vector, N::𝓝, fₕ=σ, fₒ=σ)
    N.net[1].neurons = x
    L = N.nlayers
    for i in 1:(L-2)
        W = N.net[i+1].W
        a = N.net[i].neurons
        b = N.net[i+1].biases
        z = compute_layer(W, a, b)
        N.net[i+1].neurons = fₕ(z)
    end
    # Output layer with special function
    W = N.net[L].W
    a = N.net[L-1].neurons
    b = N.net[L].biases
    z = compute_layer(W, a, b)
    N.net[L].neurons = fₒ(z)
    return(N)
end

function backprop(N::𝓝, y⃗::Vector)
    ∇𝓝 = 𝓝(N.dims)
    ∂C∂aᴸ = 2 * (N.net[N.nlayers].neurons - y⃗)
    for l in reverse(2:N.nlayers)
        ∂σ∂z = dσdx(logit.(N.net[l].neurons))
        P = hadamard(∂C∂aᴸ, ∂σ∂z)
        ∇W = kron(P, transpose(N.net[l-1].neurons))
        ∇𝓝.net[l].W = ∇W
        ∇𝓝.net[l].biases = P
        # Change the dimension of the vector appropriately
        ∂C∂aᴸ = zeros(Float32, size(N.net[l].W, 2))
        mul!(∂C∂aᴸ, transpose(N.net[l].W), P)
    end
    return ∇𝓝
end

function trainset_gradient(input_data::MNIST, network::𝓝,
    hidden_activation::Function=σ,
    output_activation::Function=σ)
    """
    Computes the average gradient for the whole training set
    for a given network.
    """
    costs = 0
    ∇N = 𝓝(network.dims)
    # Substract network parameters by themselves, so that all parameters are zero.
    minus∇N = ⊗(-1, ∇N)
    ∇N = ⊕(∇N, minus∇N)
    nsamples = length(input_data.targets)
    @time for i in 1:nsamples
        input = vec(input_data[i].features)
        network = fprop(input, network,
                                hidden_activation, output_activation)
        # Adds all weights and biases of ▿net and the gradient of the cost
        # in the network so as to compute the average at the end.s
        costs += cost(network.net[network.nlayers].neurons, input_data[i].targets)
        y = get_target_vector(input_data[i].targets)
        gradient = backprop(network, y)
        ∇N = ⊕(∇N, gradient)
    end
    # Average gradient over all training set:
    return (⊗(1 / nsamples, ∇N), costs/nsamples)
end

function train(input_data::MNIST, network::𝓝, convergence_criteria::Number, η)
    if η <= 0
        error("Cannot train with non-positive learning rate")
    end
    cost = 1000
    costs = []
    epoch = 1
    while cost > convergence_criteria
        print("Starting epoch ", string(epoch), ". Cost: ", cost)
        avg_gradient = trainset_gradient(input_data, network) 
        network = ⊕(network, ⊗(-η, avg_gradient[1]))
        cost = avg_gradient[2]
        append!(costs, cost)
        epoch += 1

    end
    return (network, costs)
end

function save_network(network::𝓝, filename="net"::String)
    open(filename * ".bin", "w") do f
        serialize(f, network)
    end
end

function load_network(filename)
    net = deserialize(filename)
    return net
end

function get_mini_batches(data::Vector, size::Int, n::Int)
    shuffled = shuffle(data)
    batches = [shuffled[1:size]]
    for i in 2:n
        append!(batches, [shuffled[(i*size+1):(i+1)*size]])
    end
    return batches
end

function get_target_vector(target::Int)::Vector
    target_vector = zeros(Float32, 10)
    target_vector[target+1] = 1
    return target_vector
end
