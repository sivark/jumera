#----------------------------------------------------------------------------
# Parsing arguments
#----------------------------------------------------------------------------

#using ArgParse
## To input FloatType and testing/production flag.
#
#settings = ArgParseSettings()
#@add_arg_table settings begin
#    #"--chi"
#    #    help = "bond dimension"
#    #    arg_type = Int64
#    #    required = true
#    #"--f32"
#    #    help = "enforce Float32 precision"
#    #    action = :store_true
#    #"--init-layers"
#    #    help = "number of transitional layers"
#    #    arg_type = Int64
#    #    default = 3
#    #"--tolerance"
#    #    help = "convergence tolerance"
#    #    arg_type = Float64
#    #    default = 1e-8
#end
#
##parsed = parse_args(ARGS,settings)
#
#@printf("chi = %1d, tolerance = %1.1e, init layers = %1d\n",parsed["chi"], parsed["tolerance"], parsed["init-layers"])
#
#if(parsed["f32"])
#    floattype = Float32
#else
#    floattype = Float64
#end
#

typealias Float Float64
@show(Float)

#----------------------------------------------------------------------------
# Training parameters
#----------------------------------------------------------------------------

#parameters_parsed = Dict(:EnergyDelta => parsed["tolerance"], :Qsweep => 12, :Qbatch => 50, :Qlayer => 4, :Qsingle => 4, :Qtop => 5);

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

using JLD

m = generate_random_MERA(INIT_LAYER_SHAPE);
println("Starting the transitional layer optimization...")
improveGraft!(improveNonSILtop,isingH, m, parameters_init)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape.jld", "m_$(INIT_LAYERS)layers", m)

## Load already optimized 7-layer MERA
#m = load("solutionMERA_3layers_(8,3,2,2)shape.jld","m_3layers")

println("Starting scale invariant layer optimization...")
improveGraft!(improveSILtop,isingH, m, parameters_init)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape_SILtop.jld", "m_$(INIT_LAYERS)layers", m)

#println(string(map((x) -> '-', collect(1:28))...))

#----------------------------------------------------------------------------
# GRAFTING AND GROWING A DEEPER MERA
#----------------------------------------------------------------------------

#growMERA!(m, LAYER_SHAPE, INIT_LAYERS)

println("\nDone for now... Over and out :-)\n")
