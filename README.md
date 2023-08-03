# MinCostSARSOP

## Example Usage
```julia
using POMDPs
using MinCostSARSOP
using POMDPModels
using ConstrainedPOMDPs

cpomdp = constrain(TigerPOMDP(), [1.0]) do s,a
    iszero(a) ? [1.0] : [0.0]
end

sol = MinCostSARSOPSolver()

pol = solve(sol, cpomdp)
```
