Base.@kwdef struct BlindLowerBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
end

function update!(pomdp::TabularCPOMDP, M::BlindLowerBound, Γ, S, A, _)
    residuals = M.residuals
    (;T,R,C,O) = pomdp
    γ = discount(pomdp)

    for a ∈ A
        αr_a = copy(M.α_tmp)
        αc_a = copy(M.α_tmp)
        T_a = T[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)
        for s ∈ S
            Vcb′ = 0.0
            Vrb′ = 0.0
            for idx ∈ nzrange(T_a, s)
                sp = rv[idx]
                p = nz[idx]
                Vrb′ += p*Γ[a].reward[sp]
                Vcb′ += p*Γ[a].cost[sp]
            end
            αr_a[s] = R[s,a] + γ*Vrb′
            αc_a[s] = C[s,a] + γ*Vcb′
        end
        res = max(bel_res(Γ[a].reward, αr_a), bel_res(Γ[a].cost, αc_a))
        residuals[a] = res
        Γ[a] = AlphaVec(αr_a, αc_a, a)
    end
    return Γ
end

function worst_state_alphas(pomdp::TabularCPOMDP, S, A)
    (;R, C, T) = pomdp
    γ = discount(pomdp)

    Γ = [AlphaVec(zeros(length(S)), zeros(length(S)), a) for a ∈ A]
    for a ∈ A
        nz = nonzeros(T[a])
        rv = rowvals(T[a])
        for s ∈ S
            rsa = R[s, a]
            csa = C[s, a]
            
            c_max = -Inf
            r_min = Inf
            for idx ∈ nzrange(T[a], s)
                sp = rv[idx]
                p = nz[idx]
                c′ = p*C[sp, a]
                r′ = p*R[sp, a]
                if c′ > c_max
                    c_max = c′
                    r_min = r′
                end
            end
            r_min = r_min=== Inf ? -Inf : r_min
            c_max = c_max===-Inf ?  Inf : c_max
            Γ[a].reward[s] = rsa + γ / (1 - γ) * r_min
            Γ[a].cost[s] = csa + γ / (1 - γ) * c_max
        end
    end
    return Γ
end

function POMDPs.solve(sol::BlindLowerBound, pomdp::TabularCPOMDP)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)

    Γ = worst_state_alphas(pomdp, S, A)
    resize!(sol.α_tmp, length(S))
    residuals = resize!(sol.residuals, length(A))

    iter = 0
    res_criterion = <(sol.bel_res)
    while iter < sol.max_iter && time() - t0 < sol.max_time
        update!(pomdp, sol, Γ, S, A, O)
        iter += 1
        all(res_criterion,residuals) && break
    end

    return Γ
end
