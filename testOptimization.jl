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

parameters_init  = Dict(:energyDelta => 1e-8, :Qsweep => 601 , :Qlayer => 5, :Qsingle => 4);
parameters_graft = Dict(:energyDelta => 1e-8, :Qsweep => 1001, :Qlayer => 3, :Qsingle => 2);
parameters_sweep = Dict(:energyDelta => 1e-8, :Qsweep => 401 , :Qlayer => 4, :Qsingle => 3);
# If we use Float32 then precision less than approximately 1e-7 is meaningless
# This will/should render :EnergyDelta check pointless -- so remove that?
# Should we optimize a multi-layer MERA with Float32 first
# and then promote it to Float64 for fine-tuning sweeps?
typealias Float Float32
@show(Float)

const LAYER_SHAPE       = (8,5,5,5,5,5,5,5,5,5)
const INIT_LAYERS       = 2
const INIT_LAYER_SHAPE  = LAYER_SHAPE[1:(INIT_LAYERS+1)]

# Print out the hyperparameters
println("Shape of init layers -- ", INIT_LAYER_SHAPE)
println("Initial optimization -- ", parameters_init)
println("Layer optimization -- ", parameters_graft)
println("Sweep optimization -- ", parameters_sweep)
println(string(map((x) -> '-', collect(1:28))...))
println()

#----------------------------------------------------------------------------
# PRE-TRAINING
#----------------------------------------------------------------------------

## Load already optimized 7-layer MERA
#    m = load("solutionMERA_7layers_(8,5,5,5,5,5,5,5)shape.jld","m_7layers")
#    improveGraft!(isingH, m, parameters_init,1)

m = generate_random_MERA(INIT_LAYER_SHAPE);
println("Starting the optimization...")
improveGraft!(isingH, m, parameters_init)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape.jld", "m_$(INIT_LAYERS)layers", m)
println(string(map((x) -> '-', collect(1:28))...))


#----------------------------------------------------------------------------
# GRAFTING AND GROWING A DEEPER MERA
#----------------------------------------------------------------------------

for lyr in (INIT_LAYERS+1):(length(LAYER_SHAPE)-1)
    println("\nNow adding layer number: ", lyr)
    exact_persite = exact_energy_persite(lyr);
    println("Not always exact per-site energy for this depth: ", exact_persite,"\n")

    newLayer = generate_random_layer(LAYER_SHAPE[lyr],LAYER_SHAPE[lyr+1])
    push!(m.levelTensors, newLayer)
    # It is tempting to guess a better initialization for the new layer.
    # Ideally, it is tempting to use the penultimate layer,
    # since they must be the same at the critical point, etc...
    # but local "gauge" freedom will probably make that quite useless

    # Improving the newly added layer and the top tensor
    energy_persite = improveGraft!(isingH, m, parameters_graft, 1)
    # It's important to iterate over the top layer several times,
    # otherwise it will wreck the lower layers when we sweep!

    energy_persite = improveGraft!(isingH, m, parameters_sweep, 2)
    #energy_persite = improveGraft!(isingH, m, parameters_sweep, 3)
    #energy_persite = improveGraft!(isingH, m, parameters_sweep, 4)

    # sweep over all layers
    energy_persite = improveGraft!(isingH, m, parameters_sweep)

    println("\nFinal energy of this optimized MERA: ", energy_persite)
    println("Not always exact per-site energy for this depth: ", exact_persite,"\n")
    println("Fractional error in our variational estimate: ", (energy_persite - exact_persite)/(exact_persite) )

    save("solutionMERA_$(lyr)layers_$(LAYER_SHAPE[1:lyr+1])shape.jld", "m_$(lyr)layers", m)
    println(string(map((x) -> '-', collect(1:28))...))
    println()
end

println("\nDone for now... Over and out :-)\n")
