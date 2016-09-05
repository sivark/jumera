println("Hello, World.")

using JLD

include("BinaryMERA.jl")
include("OptimizeMERA.jl")
include("IsingHam.jl")

println("Going to build the Ising micro hamiltonian.")
isingH, Dmax = build_H_Ising();

#----------------------------------------------------------------------------
# TRAINING HYPER-PARAMETERS
#----------------------------------------------------------------------------

parameters_init = Dict(:energyDelta => 1e-10 , :Qsweep => 2000 , :Qlayer => 3, :Qsingle => 2);
parameters_graft = Dict(:energyDelta => 1e-10, :Qsweep => 2000, :Qlayer => 3, :Qsingle => 2);
parameters_fullsweep = Dict(:energyDelta => 1e-10 , :Qsweep => 3000 , :Qlayer => 3, :Qsingle => 2);

const LAYER_SHAPE=(8,5,5,5,5,5,5,5,5,5)
const INIT_LAYERS=7
const INIT_LAYER_SHAPE=LAYER_SHAPE[1:(INIT_LAYERS+1)]

# Print out the hyperparameters
println("Shape of init layers -- ", INIT_LAYER_SHAPE)
println("Initial optimization -- ", parameters_init)
println("Layer optimization -- ", parameters_graft)
println("Sweep optimization -- ", parameters_fullsweep)
println(string(map((x) -> '-', collect(1:28))...))
println()

#----------------------------------------------------------------------------
# PRE-TRAINING
#----------------------------------------------------------------------------

## Load already optimized 6-layer MERA
m = load("solutionMERA_7layers_(8,5,5,5,5,5,5,5)shape.jld","m_7layers")

# m = generate_random_MERA(INIT_LAYER_SHAPE);
# println("Starting the optimization...")

# Below command works IFF we start with a single layer
#push!(h_layer, improveMERA!(m,isingH, Dmax, parameters_init) )
#improveMERA!(m, isingH, Dmax, parameters_init)
improveGraft!(isingH, m, parameters_init, 2)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape.jld", "m_$(INIT_LAYERS)layers", m)
println(string(map((x) -> '-', collect(1:28))...))


#----------------------------------------------------------------------------
# GRAFTING AND GROWING A DEEPER MERA
#----------------------------------------------------------------------------

for lyr in (INIT_LAYERS+1):(length(LAYER_SHAPE)-1)
    println("\nNow adding layer number: ", lyr, "\n")

    newLayer = generate_random_layer(LAYER_SHAPE[lyr],LAYER_SHAPE[lyr+1])
    push!(m.levelTensors, newLayer)
    # Improving the newly added layer and the top tensor
    # also the penultimate layer
    energy_persite = improveGraft!(isingH, m, parameters_graft, 2)

    # sweep over all layers
    energy_persite = improveGraft!(isingH, m, parameters_fullsweep)

    println("\nFinal energy of this optimized MERA: ", energy_persite)
    exact_persite = -4/pi - (pi/6)/(81*4^(lyr-1)); # including the leading finite-size correction
    println("Off from the 1/Nsq corrected answer by: ", (energy_persite - exact_persite)/(exact_persite) )

    save("solutionMERA_$(lyr)layers_$(LAYER_SHAPE[1:lyr+1])shape.jld", "m_$(lyr)layers", m)
    println(string(map((x) -> '-', collect(1:28))...))
    println()
end

println("\nDone for now... Over and out :-)\n")
