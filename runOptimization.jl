typealias Float Float64
@show(Float)

using JLD
using ArgParse
# To input FloatType and testing/production flag.

include("testParams.jl")

# Print out the hyperparameters
println("Shape of init layers -- ", INIT_LAYER_SHAPE)
println("Init  optimization -- ", parameters_init )
println("Graft optimization -- ", parameters_graft)
println("Sweep optimization -- ", parameters_sweep)
println(string(map((x) -> '-', collect(1:28))...))
println()

#----------------------------------------------------------------------------
# Including MERA code
#----------------------------------------------------------------------------

include("IsingHam.jl")
include("BinaryMERA.jl")
include("OptimizeMERA.jl")

println("Going to build the Ising micro hamiltonian.")
isingH, Dmax = build_H_Ising();


#----------------------------------------------------------------------------
# PRE-TRAINING
#----------------------------------------------------------------------------

m = generate_random_MERA(INIT_LAYER_SHAPE);
println("Starting the optimization...")
improveGraft!(improveNonSILtop,isingH, m, parameters_init)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape.jld", "m_$(INIT_LAYERS)layers", m)

## Load already optimized 7-layer MERA
#m = load("solutionMERA_3layers_(8,3,2,2)shape.jld","m_3layers")

println("Starting scale invariant optimization")
improveGraft!(improveSILtop,isingH, m, parameters_init)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape_SILtop.jld", "m_$(INIT_LAYERS)layers", m)

#println(string(map((x) -> '-', collect(1:28))...))

#----------------------------------------------------------------------------
# GRAFTING AND GROWING A DEEPER MERA
#----------------------------------------------------------------------------

#growMERA!(m, LAYER_SHAPE, INIT_LAYERS)

println("\nDone for now... Over and out :-)\n")
