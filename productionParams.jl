#----------------------------------------------------------------------------
# TRAINING HYPER-PARAMETERS: PRODUCTION
# If we use Float32 then precision less than approximately 1e-7 is meaningless
# This will/should render :EnergyDelta check pointless 
# Should we optimize a multi-layer MERA with Float32 first
# and then promote it to Float64 for fine-tuning sweeps?
#----------------------------------------------------------------------------


const CHI               = 5
const LAYER_SHAPE       = (8,fill(CHI,7)...)
const INIT_LAYERS       = 3
const INIT_LAYER_SHAPE  = LAYER_SHAPE[1:(INIT_LAYERS+1)]

# :EnergyDelta per sweep is set to 1e-8 because it will then take
# O(1000) iterations to improve the accuracy of the energy by 1e-5
parameters_init  = Dict(:EnergyDelta => 1e-8, :Qsweep => 12, :Qbatch => 50, :Qlayer => 4, :Qsingle => 4);
parameters_graft = Dict(:EnergyDelta => 1e-8, :Qsweep => 20, :Qbatch => 50, :Qlayer => 4, :Qsingle => 5);
parameters_sweep = Dict(:EnergyDelta => 1e-10, :Qsweep => 20, :Qbatch => 50, :Qlayer => 3, :Qsingle => 5);
parameters_shortsweep = Dict(:EnergyDelta => 1e-10, :Qsweep => 20, :Qbatch => 50, :Qlayer => 3, :Qsingle => 5);

