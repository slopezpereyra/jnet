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


# Mean squared error: takes vector (output layer) and computes
# mean squared error with respect to target value
square(x) = x^2
function cost(final_layer, target)::Float32
    target_vector = zeros(Float32, length(final_layer))
    target_vector[target+1] = 1
    sum(broadcast(square, final_layer .- target_vector))
end