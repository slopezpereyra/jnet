include("math.jl")
# Function definitions
using LinearAlgebra
using MLDatasets: MNIST
using Random
using Statistics

dataset_train = MNIST(:train)
dataset_test = MNIST(:test)

mutable struct Layer
    """Layer struct with named spaces
    neurons : Vector of real numbers
    weights : Matrix of real numbers
    biases : Vector of real numbers
    type : Type of the layer (input, hidden or output)."""

    neurons::Vector
    weights::Matrix
    biases::Vector
    type::String
    function Layer(neurons::Vector, weights::Matrix, biases::Vector, type::String)
        new(neurons, weights, biases, type)
    end
end

mutable struct Network
    """Network struct with namedspaces
    net : A dictionary with int/Layer pairs
    dims : Dimensino of the network
    nlayers : Number of layers in the network
    nparams : Total number of parameters (weights and biases)"""
    net::Dict{Int,Layer}
    dims::Vector
    nlayers::Int
    nparams::Int
    function Network(dims::Vector)
        structure = Dict()
        structure[length(dims)] = Layer(zeros(Float32, last(dims)), Array{Float32}(undef, 0, 0), [], "Output")
        for i in 1:(length(dims)-1)
            type = ifelse(i > 1, "Hidden", "Input")
            neurons = zeros(Float32, dims[i])
            weights = rand(Float32, dims[i+1], dims[i])
            biases = rand(Float32, dims[i+1])
            structure[i] = Layer(neurons, weights, biases, type)
        end
        nparams::Int32 = 0
        for i in 1:(length(dims)-1)
            nparams += dims[i] * dims[i+1] + dims[i+1]
        end
        new(structure, dims, length(dims), nparams)
    end
end

function compute_layer(layer::Layer, f::Function=relu)
    Y = zeros(Float32, length(layer.biases))
    Y = mul!(Y, layer.weights, layer.neurons) + layer.biases
    f(Y)
end

# Prototype. Seems to work? (See test in test sections)
function get_mini_batches(data::MNIST, size::Int, n::Int)
    shuffled = Shuffle.shuffle(data)
    batches = [shuffled[1:size]]
    for i in 2:n
        append!(batches, [shuffled[(i*size+1):(i+1)*size]])
    end
    return batches
end

function get_target_vector(target::Int)
    target_vector = zeros(Float32, 10)
    target_vector[target+1] = 1
end


function f_propagation(image::Matrix, network::Network,
    hidden_activation::Function=relu,
    output_activation::Function=softmax)
    """Perform forward propagation by defining input layer as
    determined by input x and propagating through predefined
    empty network."""
    network.net[1].neurons = vec(image)
    for i in 2:network.nlayers
        # Compute ith layer based on (i-1)nth layer's parameters and values.
        network.net[i].neurons = compute_layer(network.net[i-1], hidden_activation)
    end
    network.net[network.nlayers].neurons = output_activation(network.net[network.nlayers].neurons)
    return network
end

function train(input_data::MNIST, network::Network,
    hidden_activation::Function=relu,
    output_activation::Function=softmax,
    Δ::Number=0, Δᵢ::Tuple=(1, 1, 1))
    """Trains the network net given certain input data with the possibility of 
    using a delta displacement. By default, delta = 0 and no displacement is applied.
    input_data : MNIST dataset.
    net : A network as defined by init_network.
    Δ : Displacement value for specific parameter given by Δᵢ.
    Δᵢ (Layer, component (weight or bias), row index for element index) Layer from 1 to n-1, component is 2 o 3
    (weight and bias resepctevely), element index """
    costs = zeros(Float32, 60000)
    @time for i in 1:length(input_data.targets)
        network = f_propagation(input_data[i].features, network,
            hidden_activation, output_activation,
            Δ, Δᵢ)

        costs[i] = cost(network.net[network.nlayers].neurons, input_data[i].targets)
    end
    return mean(costs)
end


function update_parameters(network::Network, gradient::Vector, η::Float32)
    for i in 1:(network.nparams)
        network = shift_parameter(network, i, -gradient[i] * η)
    end
    return network
end

function backprop(network::Network, target)
    """
    Computes gradient of the loss function with respect to the weights
    and biases in a given network for a single sample.
    """

    ▿aᴸ = 2 * (network.net[network.nlayers].neurons - get_target_vector(target))  # Derivative of cost function with respect to activations in last layer
    ▿net = Network(network.dims)
    ▿net.net[network.nlayers].neurons = ▿aᴸ

    for l in reverse(1:(network.nlayers-1))
        dσ_dz = ddx_σ(logit(network.net[l+1].neurons)) # Derivative of the sigmoid function evaluated in z.  
        p = hadamard(▿net.net[l+1].neurons, dσ_dz)
        ▿net.net[l].biases = p
        ▿net.net[l].weights = outerproduct(dσ_dz, network.net[l].neurons)
        ▿net.net[l].neurons = mul!(▿net.net[l].neurons, tranpose(network.net[l].weights), p)
    end
    return ▿net
end