function min_alpha_val(Γ, b)
    min_α = first(Γ)
    min_val = Inf
    for α ∈ Γ
        val = dot(α.cost, b)
        if val < min_val
            min_α = α
            min_val = val
        end
    end
    return min_α
end

function backup_a!(αr, αc, pomdp::TabularCPOMDP, cache::TreeCache, a, Γao_r, Γao_c)
    γ = discount(pomdp)
    C = @view pomdp.C[:,a]
    R = @view pomdp.R[:,a]
    T_a = pomdp.T[a]
    Z_a = cache.Oᵀ[a]
    Γa_r = @view Γao_r[:,:,a]
    Γa_c = @view Γao_c[:,:,a]
    

    Tnz = nonzeros(T_a)
    Trv = rowvals(T_a)
    Znz = nonzeros(Z_a)
    Zrv = rowvals(Z_a)

    for s ∈ eachindex(α)
        vr = 0.0
        vc = 0.0
        for sp_idx ∈ nzrange(T_a, s)
            sp = Trv[sp_idx]
            p = Tnz[sp_idx]
            tmp_r = 0.0
            tmp_c = 0.0
            for o_idx ∈ nzrange(Z_a, sp)
                o = Zrv[o_idx]
                po = Znz[o_idx]
                tmp_r += po*Γa_r[sp, o]
                tmp_c += po*Γa_c[sp, o]
            end
            vr += tmp_r*p
            vc += tmp_c*p
        end
        αr[s] = vr
        αc[s] = vc
    end
    @. αr = R + γ*αr
    @. αc = C + γ*αc
    return αr, αc
end

function backup!(tree, b_idx)
    Γ = tree.Γ
    b = tree.b[b_idx]
    pomdp = tree.pomdp
    γ = discount(tree)
    S = states(tree)
    A = actions(tree)
    O = observations(tree)

    Γao = tree.cache.Γ

    for a ∈ A
        ba_idx = tree.b_children[b_idx][a]
        for o ∈ O
            bp_idx = tree.ba_children[ba_idx][o]
            bp = tree.b[bp_idx]
            _α = min_alpha_val(Γ, bp)
            Γao_r[:,o,a] .= _α.reward
            Γao_c[:,o,a] .= _α.cost
        end
    end

    Vc = Inf
    αr_a = zeros(Float64, length(S))
    αc_a = zeros(Float64, length(S))
    best_αr = zeros(Float64, length(S))
    best_αc = zeros(Float64, length(S))
    best_action = first(A)

    for a ∈ A
        αr_a, αc_a  = backup_a!(αr_a, αc_a, pomdp, tree.cache, a, Γao_r, Γao_c)
        Qcba = dot(α_a.cost, b)
        tree.Qa_lower[b_idx][a] = -Qcba
        if Qcba < Vc
            Vc = Qba
            best_αr .= αr_a
            best_αc .= αc_a
            best_action = a
        end
    end

    α = AlphaVec(best_αr, best_αc, best_action)
    push!(Γ, α)
    tree.V_lower[b_idx] = -Vc
end

function backup!(tree)
    for i ∈ reverse(eachindex(tree.sampled))
        backup!(tree, tree.sampled[i])
    end
end
