typealias Float Float64
@show(Float)

include("trainingParams.jl")

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

using JLD

#----------------------------------------------------------------------------
# PRE-TRAINING
#----------------------------------------------------------------------------

## Load already optimized 7-layer MERA
#    m = load("solutionMERA_7layers_(8,5,5,5,5,5,5,5)shape.jld","m_7layers")
#    improveGraft!(isingH, m, parameters_init,1)

m = generate_random_MERA(INIT_LAYER_SHAPE);
println("Starting the optimization...")
improveGraft!(isingH, m, parameters_init)
# Not storing rhoslist to disk, for INIT_LAYERS
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape.jld", "m_$(INIT_LAYERS)layers", m)
#println(string(map((x) -> '-', collect(1:28))...))


#----------------------------------------------------------------------------
# GRAFTING AND GROWING A DEEPER MERA
#----------------------------------------------------------------------------

for lyr in (INIT_LAYERS+1):(length(LAYER_SHAPE)-1)
    println("\nNow adding layer number: %2d of bond dimensions %2d -> %2d", lyr, LAYER_SHAPE[lyr], LAYER_SHAPE[lyr+1])
    exact_persite = exact_energy_persite(lyr);
    println("Not always exact per-site energy for this depth: ", exact_persite,"\n")

    #newLayer = generate_random_layer(LAYER_SHAPE[lyr],LAYER_SHAPE[lyr+1])
    push!(m.levelTensors, m.topLayer.levelTensors)
    m.topLayer = generate_random_top(LAYER_SHAPE[lyr],LAYER_SHAPE[lyr+1])

    # It is tempting to guess a better initialization for the new layer.
    # Ideally, it is tempting to use the penultimate layer,
    # since they must be the same at the critical point, etc...
    # but local "gauge" freedom will probably make that quite useless

    jldopen("rhoslist_snapshots_$(length(m.levelTensors))layers.jld","w") do file
        # Improving the newly added layer and the top tensor
        rhoslist_snapshots1 = improveGraft!(isingH, m, parameters_graft, 1)
        # It's important to iterate over the top layer several times,
        # otherwise it will wreck the lower layers when we sweep!
        write(file, "rhoslist_snapshots_1smoothing", rhoslist_snapshots1)

        rhoslist_snapshots2 = improveGraft!(isingH, m, parameters_sweep, 2)
        write(file, "rhoslist_snapshots_2smoothing", rhoslist_snapshots2)

        #rhoslist_snapshots3 = improveGraft!(isingH, m, parameters_shortsweep, 3)
        #write(file, "rhoslist_snapshots_3smoothing", rhoslist_snapshots3)

        # sweep over all layers
        rhoslist_snapshotsAll = improveGraft!(isingH, m, parameters_sweep)
        write(file, "rhoslist_snapshots_$(lyr)smoothing", rhoslist_snapshotsAll)
    end

    # println("\nFinal energy of this optimized MERA: ", energy_persite)
    # println("Not always exact per-site energy for this depth: ", exact_persite,"\n")
    # println("Fractional error in our variational estimate: ", (energy_persite - exact_persite)/(exact_persite) )
    save("solutionMERA_$(lyr)layers_$(LAYER_SHAPE[1:lyr+1])shape.jld", "m_$(lyr)layers", m)
    println(string(map((x) -> '-', collect(1:28))...))
    println()
end

println("\nDone for now... Over and out :-)\n")
