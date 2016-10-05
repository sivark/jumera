#----------------------------------------------------------------------------
# TRAINING HYPER-PARAMETERS: TESTING
#----------------------------------------------------------------------------

const CHI               = 5
const LAYER_SHAPE       = (8,3,2,4,3,4,2)
const INIT_LAYERS       = 3
const INIT_LAYER_SHAPE  = LAYER_SHAPE[1:(INIT_LAYERS+1)]

# :EnergyDelta per sweep is set to 1e-8 because it will then take
# O(1000) iterations to improve the accuracy of the energy by 1e-5
parameters_init  = Dict(:EnergyDelta => 1e-4, :Qsweep => 12, :Qbatch => 5, :Qlayer => 4, :Qsingle => 4, :Qtop => 5);
parameters_graft = Dict(:EnergyDelta => 1e-4, :Qsweep => 2, :Qbatch => 5, :Qlayer => 4, :Qsingle => 5, :Qtop => 5);
parameters_sweep = Dict(:EnergyDelta => 1e-4, :Qsweep => 2, :Qbatch => 5, :Qlayer => 3, :Qsingle => 5, :Qtop => 5);
parameters_shortsweep = Dict(:EnergyDelta => 1e-5, :Qsweep => 2, :Qbatch => 5, :Qlayer => 3, :Qsingle => 5, :Qtop => 5);
