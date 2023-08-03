using MinCostSARSOP
using ConstrainedPOMDPs
using ConstrainedPOMDPModels
using POMDPs
using POMDPTools
using Test

@testset "tiger" begin
    cpomdp = constrain(TigerPOMDP(), [1.0]) do s,a
        iszero(a) ? [1.0] : [0.0]
    end
    
    sol = MinCostSARSOPSolver()
    pol = solve(sol, cpomdp)
    
    @test value(pol, initialstate(cpomdp)) ≈ 0.0
end
