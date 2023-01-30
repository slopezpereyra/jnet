# Networks and layers; network and layer operators.

using LinearAlgebra

mutable struct 𝓛 
    
    neurons::Vector 
    W::Matrix
    biases::Vector

    function 𝓛(neurons::Vector, W::Matrix, biases::Vector)
        new(neurons, W, biases)
    end
end

mutable struct 𝓝
    net::Dict{Int, 𝓛}
    dims::Vector
    nlayers::Int
    nparams::Int

    function 𝓝(dims::Vector)
        structure = Dict()
        structure[1] = 𝓛(zeros(Float32, dims[1]), Array{Float32}(undef, 0, 0), [])
        for i in 2:length(dims)
            neurons = zeros(Float32, dims[i])
            weights = rand(Float32, dims[i], dims[i-1])
            biases = rand(Float32, dims[i])
            structure[i] = 𝓛(neurons, weights, biases)
         end
        n::Int32 = 0
        for i in 2:length(dims)
            n += dims[i - 1] * dims[i] + dims[i]
        end

        new(structure, dims, length(dims), n)
    end 
end

function ⊕(L₁::𝓛, L₂::𝓛)
    """Addition operator under over the set 𝑳 of layer objects.
    Observe that (𝑳, ⊕) is a non-abbelian group. In particular,
    the operation

                    ℒ₁ ⊕ ℒ₂ = ℒ₃,          ℒᵢ∈ 𝑳

    is neuron-preserving with respect to ℒ₁. This is, 
    ℒ₃ has the same activations as ℒ₁"""

    L₃ = 𝓛(L₁.neurons, L₁.W + L₂.W, L₁.biases + L₂.biases)
    return L₃
end

function ⊗(λ::Number, L::𝓛)
    """Scalar-layer multiplication.
    ⊗ is the group action

                ⊗ : ℝ × 𝑳 → 𝑳 
    """

    W = λ * L.W
    b = λ * L.biases
    return 𝓛(L.neurons, W, b)
end

function ⊗(λ::Number, N::𝓝)
    """Scalar-network multiplication ⊗.
    Might be thought of as a group action 

                ⊗ : ℝ × 𝑵 → 𝑵

    This operation is replacing. """

    new = 𝓝(N.dims)
    keys = [1:N.nlayers;]
    layers = [⊗(λ, N.net[i]) for i in 1:N.nlayers]
    new.net = Dict(keys .=> layers)
    return new
end

function ⊕(N₁::𝓝, N₂::𝓝)
    """Addition operator over the set 𝑵 of layer objects.
    Observe that (𝑵, ⊕) is a non-abbelian group. In particular, 

                    𝐧₁ ⊕ 𝐧₂ = 𝐧₃           𝐧ᵢ∈ 𝑵

    is neuron-preserving with respect to 𝐧₁."""

    if N₁.dims != N₂.dims
        throw(DimensionMismatch)
    end
    N₃ = 𝓝(N₁.dims)
    N₃.net = mergewith(⊕, N₁.net, N₂.net)
    return N₃
end

#function gradient_norm(▿net::𝓝)
#    #Modificarla para que concuerde con las funciones de Santi
#    norm = 0 #Float32
#    for l in 1:(▿net.nlayers-1)
#        norm += norm(network.net[l].weights)^2 + norm(network.net[l].biases)^2
#    end
#    sqrt(norm)
#end

