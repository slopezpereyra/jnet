# Mathematical functions

relu(x::Number) = max(0, x)
function relu(X::Vector)
    relu.(X)
end

function softmax(X::Vector)
    X = X .- maximum(X)
    exp.(X) ./ sum(exp.(X))
end

σ(x::Number) = 1 / (1 + exp(-x))
function σ(X::Vector)
    σ.(X)
end

function dσdx(x::Number)
    σ(x)*(1 - σ(x))
end

function logit(x)
    if x < 0 || x > 1
        error("Logit input out of bounds")
    end
    log(x / (1 - x))
end


function dσdx(X::Vector)
    broadcast(dσdx, X)
end

function dreludx(x::Number)
    ifelse(x > 0, 1, 0)
end

square(x) = x^2
function cost(final_layer, target)::Float32
    target_vector = zeros(Float32, length(final_layer))
    target_vector[target+1] = 1
    sum(broadcast(square, final_layer .- target_vector))
end

# function ddx_cost(x::Number, y::Number)::Number
#     2*(x - y)
# end

# function ddx_cost(x⃗)

# Hadamard product
function hadamard(A, B)
    C = broadcast(*, A, B)
    return C
end

# ERROR: This function returns Matrix{Matrix{Float32}} type, not {Matrix{Float32}}, which is strange.
function outerproduct(A, B)
    [r * c' for r in eachrow(A), c in eachcol(B)]
end
