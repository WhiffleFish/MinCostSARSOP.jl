Base.@kwdef struct FastInformedBound <: Solver
    max_iter::Int               = typemax(Int)
    max_time::Float64           = 1.
    bel_res::Float64            = 1e-3
    init_value::Float64         = 0.
    α_tmp::Vector{Float64}      = Float64[]
    residuals::Vector{Float64}  = Float64[]
end

function bel_res(α1, α2)
    max_res = 0.
    @inbounds for i ∈ eachindex(α1, α2)
        res = abs(α1[i] - α2[i])
        res > max_res && (max_res = res)
    end
    return max_res
end

function update!(𝒫::TabularCPOMDP, M::FastInformedBound, Γ, 𝒮, 𝒜, 𝒪)
    (;C,R,T,O) = 𝒫
    γ = discount(𝒫)
    residuals = M.residuals

    for a ∈ 𝒜
        αr_a = copy(M.α_tmp) # TODO: actually use the cache
        αc_a = copy(M.α_tmp)
        T_a = T[a]
        O_a = O[a]
        nz = nonzeros(T_a)
        rv = rowvals(T_a)

        for s ∈ 𝒮
            rsa = R[s,a]
            csa = C[s,a]

            if isinf(csa)
                αr_a[s] = -Inf
                αc_a[s] = Inf
            elseif isterminal(𝒫,s)
                αr_a[s] = 0.
                αc_a[s] = 0.
            else
                tmp_c = 0.0
                tmp_r = 0.0
                for o ∈ 𝒪
                    O_ao = @view O_a[:,o] # FIXME: slow sparse indexing for inner O_ao[sp]
                    Vc_opt = Inf
                    Vr_opt = -Inf
                    for α′ ∈ Γ
                        Vcb′ = 0.0
                        Vrb′ = 0.0
                        for idx ∈ nzrange(T_a, s)
                            sp = rv[idx]
                            Tprob = nz[idx]
                            Vcb′ += O_ao[sp]*Tprob*α′.cost[sp]
                            Vrb′ += O_ao[sp]*Tprob*α′.reward[sp]
                        end
                        if Vcb′ < Vc_opt
                            Vc_opt = Vcb′
                            Vr_opt = Vrb′
                        end
                    end
                    tmp_c += Vc_opt
                    tmp_r += Vr_opt
                end
                αc_a[s] = csa + γ*tmp_c
                αr_a[s] = rsa + γ*tmp_r
            end
        end
        res = max(bel_res(Γ[a].reward, αr_a), bel_res(Γ[a].cost, αc_a))
        residuals[a] = res
        Γ[a] = AlphaVec(αr_a, αc_a, a)
    end
end

function POMDPs.solve(sol::FastInformedBound, pomdp::TabularCPOMDP)
    t0 = time()
    S = ordered_states(pomdp)
    A = ordered_actions(pomdp)
    O = ordered_observations(pomdp)
    γ = discount(pomdp)

    init_value = sol.init_value
    Γ = if isfinite(init_value)
        [AlphaVec(fill(init_value, length(S)), fill(init_value, length(S)), a) for a ∈ A]
    else
        c_min = maximum(only(costs(pomdp, s, a)) for a ∈ A, s ∈ S)
        r_max = maximum(reward(pomdp, s, a) for a ∈ A, s ∈ S)
        V̄c = c_min/(1-γ)
        V̄r = r_max/(1-γ)
        [AlphaVec(fill(V̄r, length(S)), fill(V̄c, length(S)), a) for a ∈ A]
    end
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
