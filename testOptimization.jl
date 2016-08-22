# Ising model with transverse magnetic field h (critical h=1 by default)
println("Hello World!")

using JLD

include("BinaryMERA.jl")
include("OptimizeMERA.jl")

println("Going to build the Ising micro hamiltonian.")

CHI = 8
INIT_LAYERS = 4

function build_H_Ising(h=1.0)
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
    H = reshape(H, (8,8,8,8,8,8))
    return H, D_max
end

isingH, Dmax = build_H_Ising();
parameters_1 = Dict(:energyDelta => 0.0001 , :maxIter => 5 , :layerIters => 4)

m = generate_random_MERA(CHI,INIT_LAYERS);
improveMERA!(m,isingH,parameters)
save("solutionMERA_testTranspose_chi$(CHI)_4layers.jld", "m_4layers", m)

## Already optimized 5-layer MERA
#m = load("solutionMERA_chi8_5layers.jld","m_5layers")

println("Starting the optimization...\n")

#h_layer = ascendTo(complex(isingH), m, 4);

## Growing a deep MERA by pre-training lower layers
#for lyr in (INIT_LAYERS+1):10
#    println("\nNow adding layer number: ", lyr, "\n")
#
#    # Add a new layer, keeping the toptensor the same!
#    newLayer = generate_random_layer(CHI)
#    parameters_newlayer = Dict(:layerIters => 2^6)
#    newLayer = improveLayer(h_layer, newLayer, descendTo(m,lyr-1), parameters_newlayer)
#    push!(m.levelTensors, newLayer)
#
#    # Open a logging file for this layer so that all the outputs could be piped to that
#
#    # sweep over all layers
#    parameters_fullsweep = Dict(:energyDelta => 1e-10 , :maxIter => 100*(2^lyr) , :layerIters => 2)
#    h_layer = improveMERA!(m,isingH,parameters_fullsweep)
#
#    save("solutionMERA_chi$(CHI)_$(lyr)layers.jld", "m_$(lyr)layers", m)
#end

println("\nDone for the night... Over and out :-)\n")
