# Ising model with transverse magnetic field h (critical h=1 by default)
println("Hello, World.")

using JLD

include("BinaryMERA.jl")
include("OptimizeMERA.jl")

println("Going to build the Ising micro hamiltonian.")

function build_H_Ising(h=1.0)
    h::Array{Complex{Float64},6}
    D_max::Float64
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    I = eye(2)
    XX = kron(X,X)
    ZI = kron(Z,I)
    IZ = kron(I,Z)
    H2 = -(XX + h/2*(ZI+IZ))
    H = H2 / 3  # See below for an explanation of the 1/3.
    for n = 3:9
        eyen2 = eye(2^(n-2))
        # Terms at the borders of the blocks of three that get grouped together
        # need to be normalized differently from the ones that are within blocks.
        factor = (n == 4 || n == 7) ? 1/2 : 1/3
        H = kron(H, I) + kron(eyen2, H2)*factor
    end
    D, V = eig(Hermitian(H))
    D_max = D[end]
    # subtract largest eigenvalue, so that the spectrum is negative
    H = H - eye(2^9)*D_max
    h = reshape(H, (8,8,8,8,8,8)) |> complex
    return h, D_max
end

isingH, Dmax = build_H_Ising();


#----------------------------------------------------------------------------
# TRAINING HYPER-PARAMETERS
#----------------------------------------------------------------------------

parameters_init = Dict(:energyDelta => 1e-10 , :Qsweep => 500 , :Qlayer => 3, :Qsingle => 2);
parameters_graft = Dict(:energyDelta => 1e-10, :Qsweep => 500, :Qlayer => 3, :Qsingle => 2);
# The previous layers went through 1000 iterations -- what right do we have to make the new ones go through less!?
# Only after these many will the error range of the new one also come down to 10^-6
parameters_fullsweep = Dict(:energyDelta => 1e-10 , :Qsweep => 500 , :Qlayer => 3, :Qsingle => 2);

# 8 layers, each of BD 5
const LAYER_SHAPE=(8,5,5,5,5,5,5,5,5)
const INIT_LAYERS=3
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

m = generate_random_MERA(INIT_LAYER_SHAPE);
println("Starting the optimization...")

# Below command works IFF we start with a single layer
#push!(h_layer, improveMERA!(m,isingH, Dmax, parameters_init) )
#improveMERA!(m, isingH, Dmax, parameters_init)
improveGraft!(isingH, m, parameters_init)
save("solutionMERA_$(INIT_LAYERS)layers_$(INIT_LAYER_SHAPE)shape.jld", "m_$(INIT_LAYERS)layers", m)
println(string(map((x) -> '-', collect(1:28))...))

## Load already optimized 5-layer MERA
#m = load("solutionMERA_chi8_5layers.jld","m_5layers")

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
    exact_persite = -4/pi + (0.12928)/(2^(lyr*2)); # including the leading finite-size correction
    println("Off from the exact answer by: ", (energy_persite - exact_persite)/(exact_persite) )

    save("solutionMERA_$(lyr)layers_$(LAYER_SHAPE[1:lyr+1])shape.jld", "m_$(lyr)layers", m)
    println(string(map((x) -> '-', collect(1:28))...))
    println()
end

println("\nDone for now... Over and out :-)\n")
