typealias Float Float64
using Plots
pyplot()
include("IsingHam.jl")
include("BinaryMERA.jl")
include("OptimizeMERA.jl")
include("MakePlots.jl")

cd("./r16/")
@show currdir=pwd()

animatedplot("entanglement_evolution.gif";n_start=3,n_stop=9,loop=1)
