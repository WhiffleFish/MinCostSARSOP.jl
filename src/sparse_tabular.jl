struct TabularCPOMDP <: POMDP{Int,Int,Int}
    T::Vector{SparseMatrixCSC{Float64, Int64}} # T[a][sp, s]
    R::Array{Float64, 2} # R[s,a]
    O::Vector{SparseMatrixCSC{Float64, Int64}} # O[a][sp, o]
    C::Matrix{Float64} # C[s,a]
    isterminal::BitVector
    initialstate::SparseVector{Float64, Int}
    constraints::Vector{Float64}
    discount::Float64
end

function TabularCPOMDP(pomdp::CPOMDP)
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    T = _tabular_transitions(pomdp, S, A)
    R = _tabular_rewards(pomdp, S, A)
    O = _tabular_observations(pomdp, S, A, O)
    C = _tabular_costs(pomdp, S, A)
    term = _vectorized_terminal(pomdp, S)
    b0 = _vectorized_initialstate(pomdp, S)
    return TabularCPOMDP(T,R,O,C,term,b0,pomdp.constraints,discount(pomdp))
end

function _tabular_transitions(pomdp, S, A)
    T = [Matrix{Float64}(undef, length(S), length(S)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(T)
        _fill_transitions!(pomdp, T[i], S, A[i])
    end
    T
end

function _fill_transitions!(pomdp, T, S, a)
    for (s_idx, s) ∈ enumerate(S)
        Tsa = transition(pomdp, s, a)
        for (sp_idx, sp) ∈ enumerate(S)
            T[sp_idx, s_idx] = pdf(Tsa, sp)
        end
    end
    T
end

function _tabular_rewards(pomdp, S, A)
    R = Matrix{Float64}(undef, length(S), length(A))
    for (s_idx, s) ∈ enumerate(S)
        for (a_idx, a) ∈ enumerate(A)
            R[s_idx, a_idx] = reward(pomdp, s, a)
        end
    end
    R
end

function _tabular_observations(pomdp, S, A, O)
    _O = [Matrix{Float64}(undef, length(S), length(O)) for _ ∈ eachindex(A)]
    for i ∈ eachindex(_O)
        _fill_observations!(pomdp, _O[i], S, A[i], O)
    end
    _O
end

function _fill_observations!(pomdp, Oa, S, a, O)
    for (sp_idx, sp) ∈ enumerate(S)
        obs_dist = observation(pomdp, a, sp)
        for (o_idx, o) ∈ enumerate(O)
            Oa[sp_idx, o_idx] = pdf(obs_dist, o)
        end
    end
    Oa
end

function _tabular_costs(pomdp, S, A)
    @assert isone(ConstrainedPOMDPs.constraint_size(pomdp))
    C = Matrix{Float64}(undef, length(S), length(A))
    for (s_idx,s) ∈ enumerate(S)
        for (a_idx,a) ∈ enumerate(A)
            C[s_idx, a_idx] = only(costs(pomdp, s, a))
        end
    end
    C
end

function _vectorized_terminal(pomdp, S)
    term = BitVector(undef, length(S))
    @inbounds for i ∈ eachindex(term,S)
        term[i] = isterminal(pomdp, S[i])
    end
    return term
end

function _vectorized_initialstate(pomdp, S)
    b0 = initialstate(pomdp)
    b0_vec = Vector{Float64}(undef, length(S))
    @inbounds for i ∈ eachindex(S, b0_vec)
        b0_vec[i] = pdf(b0, S[i])
    end
    return sparse(b0_vec)
end

POMDPTools.ordered_states(pomdp::TabularCPOMDP) = axes(pomdp.R, 1)
POMDPs.states(pomdp::TabularCPOMDP) = ordered_states(pomdp)
POMDPTools.ordered_actions(pomdp::TabularCPOMDP) = eachindex(pomdp.T)
POMDPs.actions(pomdp::TabularCPOMDP) = ordered_actions(pomdp)
POMDPTools.ordered_observations(pomdp::TabularCPOMDP) = axes(first(pomdp.O), 2)
POMDPs.observations(pomdp::TabularCPOMDP) = ordered_observations(pomdp)

POMDPs.discount(pomdp::TabularCPOMDP) = pomdp.discount

ConstrainedPOMDPs.constraint_size(pomdp::TabularCPOMDP) = size(pomdp.C, 3)

n_states(pomdp::TabularCPOMDP) = length(states(pomdp))
n_actions(pomdp::TabularCPOMDP) = length(actions(pomdp))
n_observations(pomdp::TabularCPOMDP) = length(observations(pomdp))
n_constraints(pomdp::TabularCPOMDP) = size(pomdp.C, 3)
const n_cost = n_constraints
