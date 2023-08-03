begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using NativeSARSOP
    using POMDPs
    Pkg.activate(@__DIR__)
    using ConstrainedPOMDPModels
end
    
pomdp = ModMiniHall()
sol = SARSOPSolver(delta=0.20)
solve(sol,pomdp)

@time solve(sol,pomdp)
