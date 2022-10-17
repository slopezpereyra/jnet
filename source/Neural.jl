include("math.jl")
# Function definitions
using Random
using LinearAlgebra

mutable struct Layer
    neurons::Vector
    weights::Matrix
    biases::Vector
    type::String
    function Layer(neurons::Vector, weights::Matrix, biases::Vector, type::String)
        new(neurons, weights, biases, type)
    end
end

mutable struct Network
    net::Dict{Int,Layer}
    dims::Vector
    nlayers::Int
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
        new(structure, dims, length(dims))
    end
end

# function compute_layer(weights, v, b, f::Function=relu)
#     Y = zeros(Float32, length(b))
#     Y = mul!(Y, weights, v)
#     f(Y)
# end

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

function f_propagation(image::Matrix, network::Network,
    hidden_activation::Function=relu,
    output_activation::Function=softmax,
    Δ::Number=0.0, Δᵢ::Tuple=(1, 1, 1))
    """Perform forward propagation by defining input layer as
    determined by input x and propagating through predefined
    empty network."""
    # Set input layer
    # network.net[Δᵢ[1]][Δᵢ[2]][Δᵢ[3]] += Δ
    network.net[1].neurons = vec(image)
    for i in 2:network.nlayers
        # Compute ith layer based on (i-1)nth layer's parameters and values.
        network.net[i].neurons = compute_layer(network.net[i-1], hidden_activation)
    end
    print("End loop")
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


function gradient_calculation(net::Dict, gradients::Dict)
    # ...
end

function update_parameters(network::Network, gradient::Dict, η::Float32)
    for i in 1:(network.nlayers-1)
        network.net[i].weights += -gradient[i][1] * η
        network.net[i].biases += -gradient[i][2] * η
    end
end


N = Network([784, 16, 16, 10])
N.net[1].weights
false_image = rand(Float32, 28, 28)
net = f_propagation(false_image, N)

N.net[3].neurons
