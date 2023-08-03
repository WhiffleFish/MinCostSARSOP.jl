struct AlphaVec <: AbstractVector{Float64}
    reward::Vector{Float64}
    cost::Vector{Float64} # TODO: find a way to generalize to matrix
    action::Int
end

@inline Base.length(v::AlphaVec) = length(v.cost)

@inline Base.size(v::AlphaVec) = size(v.cost)

@inline Base.getindex(v::AlphaVec, i) = -v.cost[i]
