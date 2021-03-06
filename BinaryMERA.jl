#----------------------------------------------------------------------------------------------------
# [TODO] Implement invariants for the MERA
# Can these be encoded in traits/interfaces/protocols?
#   1. Unitarity properties of U and V
#   2. Connections U -> V within a MERA layer
#   3. Asop and Dsop are adjoints (inverses as far as TN diagrams go)
#----------------------------------------------------------------------------------------------------


#module BinaryMERA
# Use as importall BinaryMERA

# -------------------------------------------------------------
# This MERA has 2->1 coarse-graining and 3-site local operators
# -------------------------------------------------------------

using TensorFactorizations
using TensorOperations
using NCon

# ------------------------------------------------------------
# DATA STRUCTURE
# ------------------------------------------------------------

type Tensor
    # ensure that this is a tensor with upper and lower indices,
    # specified shape/dimensions
    # specify contraction rules, index convention, etc.
end

typealias LocalOperator Array{Complex{Float},6}

immutable Isometry
    elem::Array{Complex{Float},3}
end

immutable Disentangler
    elem::Array{Complex{Float},4}
end

# can we separate bonds pointing to the UV and the IR, to minimize chances of contraction mistakes?
# such as primed indices representing the IR, and we can only contract a primed index with an unprimed index :-?
# And complex conjugation changes what's primed and unprimed -- then things will go through fine.
# basically upper-vs-lower indices

# ------------------------------------------------------------
# Intermediate Layers
# Each layer must provide methods to: Ascend/Descend an operator through the layer
# ------------------------------------------------------------

immutable Layer
    u::Disentangler
    w::Isometry
    udag::Disentangler
    wdag::Isometry

    # It seems more efficient to store udag and vdag once for each layer
    # rather than compute for each usage of the superoperator

    function build_udg(u::Disentangler)
        udg = Disentangler(permutedims(conj(u.elem), (3,4,1,2)))
        return udg
    end

    function build_wdg(w::Isometry)
        wdg = Isometry(permutedims(conj(w.elem), (2,3,1)))
        return wdg
    end

    Layer(u,w) = new( u, w, build_udg(u), build_wdg(w)  )
    Layer(u,w,udag,wdag) = error("Specify only the disentangler and the isometry!\n")
    # Constructor enforces the invariance IFF type is immutable
    # else it trusts the user to not do bad things like directly access udag and change it to be incompatible with u
    # Keeping structure mutable leaves the option of using it while optimizing a MERA, etc
end

# ------------------------------------------------------------
# BUILDING SUPER-OPERATORS
# Note that the rightside operator is the parity flip rotation of the leftside operator :-)
# ------------------------------------------------------------

function ascend_threesite_left(op::LocalOperator, l::Layer)
    local scaled_op::LocalOperator
    scaled_op = ncon((l.w.elem, l.w.elem, l.w.elem,
    l.u.elem, l.u.elem,
    op,
    l.udag.elem, l.udag.elem,
    l.wdag.elem, l.wdag.elem, l.wdag.elem),
    ([-100,5,6], [-200,9,8], [-300,16,15],
    [6,9,1,2], [8,16,10,12],
    [1,2,10,3,4,14],
    [3,4,7,13], [14,12,11,17],
    [5,7,-400], [13,11,-500], [17,15,-600]))
    return scaled_op
end

function ascend_threesite_right(op::LocalOperator, l::Layer)
    local scaled_op::LocalOperator
    scaled_op = ncon((l.w.elem, l.w.elem, l.w.elem,
    l.u.elem, l.u.elem,
    op,
    l.udag.elem, l.udag.elem,
    l.wdag.elem, l.wdag.elem, l.wdag.elem),
    ([-100,15,16], [-200,8,9], [-300,6,5],
    [16,8,12,10], [9,6,2,1],
    [10,2,1,14,4,3],
    [12,14,17,11], [4,3,13,7],
    [15,17,-400], [11,13,-500], [7,5,-600]))
    return scaled_op
end

function ascend_threesite_symm(op::LocalOperator, l::Layer)
    return convert(Float,0.5)*( ascend_threesite_left(op,l)+ascend_threesite_right(op,l) )
end

function descend_threesite_right(op::LocalOperator, l::Layer)
    local scaled_op::LocalOperator
    scaled_op = ncon((l.wdag.elem, l.wdag.elem, l.wdag.elem,
    l.udag.elem, l.udag.elem,
    op,
    l.u.elem, l.u.elem,
    l.w.elem, l.w.elem, l.w.elem),
    ([4,9,5],[7,17,8],[16,1,2],
    [13,-100,9,7],[-200,-300,17,16],
    [5,8,2,6,11,3],
    [12,10,13,-400], [15,14,-500,-600],
    [6,4,12], [11,10,15], [3,14,1]))
    return scaled_op
end

function descend_threesite_left(op::LocalOperator, l::Layer)
    local scaled_op::LocalOperator
    scaled_op = ncon((l.wdag.elem, l.wdag.elem, l.wdag.elem,
    l.udag.elem, l.udag.elem,
    op,
    l.u.elem, l.u.elem,
    l.w.elem, l.w.elem, l.w.elem),
    ([1,16,2],[17,7,8],[9,4,5],
    [-100,-200,16,17],[-300,13,7,9],
    [2,8,5,3,11,6],
    [14,15,-400,-500],[10,12,-600,13],
    [3,1,14],[11,15,10],[6,12,4]))
    return scaled_op
end

function descend_threesite_symm(op::LocalOperator, l::Layer)
    return convert(Float,0.5)*( descend_threesite_left(op,l)+descend_threesite_right(op,l) )
end

function Asop(l::Layer)
    # Return linear map on op::LocalOperator ??
    # Convenient for tensor operations and for finding fixedpoint!
    return (   op -> ascend_threesite_symm(op, l)    )
end

function Dsop(l::Layer)
    # Return linear map on op::LocalOperator ??
    # Convenient for tensor operations and for finding fixedpoint!
    return (   op -> descend_threesite_symm(op, l)    )
end

# ------------------------------------------------------------
# Top Layer
# ------------------------------------------------------------


#abstract TopLayer
# This is the parent abstract type
# All daughters must export methods to:
# 1. Supply a state for the later below
# 2. Use that state to evaluate and return energy
# 3. Export a method to train the top layer

immutable TopLayer
    levelTensors::Layer
    state::LocalOperator # 3-site RDM
end

# immutable SILtop <: TopLayer
#     levelTensors::Layer
#     state::LocalOperator # 3-site RDM
# end
#
# immutable nonSILtop <: TopLayer
#     levelTensors::Layer
#     state::LocalOperator # 3-site RDM
# end

# function generate_random_SILtop(chi)
#     levelTensors = generate_random_layer(chi_lower,chi_upper)
#     state = fixedpoint(idOp, levelTensors)
#
#     return nonSILtop(levelTensors,state)
# end
#
# function generate_random_nonSILtop(chi_lower,chi_upper)
#     levelTensors::Layer
#     state::LocalOperator
#
#     top= randn(ntuple(_ -> chi, 3)...)
#     top /= vecnorm(top)
#
#     levelTensors = generate_random_layer(chi_lower,chi_upper)
#     state = dm4pureState(top)
#
#     return nonSILtop(levelTensors,state)
# end

function getTopState(t::TopLayer)
    return descend_threesite_symm(t.state, t.levelTensors)
end

function generate_random_top(chi_lower,chi_upper)
    local levelTensors::Layer
    local state::LocalOperator

    top= randn(ntuple(_ -> chi_upper, 3)...)
    top /= vecnorm(top)

    levelTensors = generate_random_layer(chi_lower,chi_upper)
    state = dm4pureState(top)

    return TopLayer(levelTensors,state)
end

#Should I generate a "random" state in the constructor and then compute
#the fixed-point of the tensors when I need to fetch the state (lazy evaluation)
#or should I do it eagerly right when I construct the tensors?
#If I construct it eagerly and store it in the object, then I don't really need
#a getState() method.
#

# ------------------------------------------------------------
# MERA data type
# Organize MERA as a composition of layers.
# ------------------------------------------------------------


type MERA
    # MUTABLE container class
    # since we can optimize in-place and save memory allocation
    levelTensors::Array{Layer} # sequence of layers
    #topTensor::Array{Complex{Float},3} # 3 indices
    topLayer::TopLayer
end
# Is it okay for concrete type MERA to be composed of an abstract type TopLayer? Not okay!
# Otherwise how will the constructor instantiate a MERA?!


function ascendTo(op::LocalOperator,m::MERA,EvalScale::Int64)
    opAtEvalScale = op
    for i in collect(1:EvalScale)
        opAtEvalScale = ascend_threesite_symm(opAtEvalScale,m.levelTensors[i])
    end
    # Haven't accounted for the possible stepping up once through the top layer
    return opAtEvalScale
end

function descendTo(m::MERA,EvalScale::Int)
    # evalscale starts at zero below layer1
    totLayers = length(m.levelTensors)
    stateAtEvalScale = dm4pureState(m.topTensor)
    for j in reverse((EvalScale+1):totLayers)
        stateAtEvalScale = descend_threesite_symm(stateAtEvalScale,m.levelTensors[j])
    end
    return stateAtEvalScale
end

function expectation(op::Array{Complex{Float64},6},rho::Array{Complex{Float64},6})
    # Need operator and rho to be given at the same scale
    # Scale for operators is 1-indexed while
    # Scale for states is 0-indexed at the ultraviolet cutoff
    result = ncon((op,rho),([1,2,3,4,5,6],[4,5,6,1,2,3]))[1]
    # NCon returns a zero dimensional array, and we're pulling out the first element
    return result
end

function generate_random_MERA(listOfChis)
    uw_list = []
    for i in 1:(length(listOfChis)-2)
        push!(uw_list, generate_random_layer(listOfChis[i],listOfChis[i+1]) )
    end
    topTensor = generate_random_top(listOfChis[end-1],listOfChis[end])

    return MERA(uw_list, topTensor)
end


# ------------------------------------------------------------
# UTILITY FUNCTIONS and DEFINITIONS
# ------------------------------------------------------------

threesiteeye(chi) = complex(ncon((eye(chi),eye(chi),eye(chi)),([-1,-11], [-2,-12], [-3,-13])));

function random_complex_tensor(chi, rank)
    local res::Array{Complex{Float},rank}
    real = randn(ntuple(_ -> chi, rank)...)
    imag = randn(ntuple(_ -> chi, rank)...)
    res = real + im*imag
    return res
end

function generate_random_layer(chi_lower,chi_upper)
    # first disentangle and then coarsegrain
    # so the bond-dimension given out by U must match the bond dimension taken in by W

    # Generate a random tensor and SVD it to get a "random" unitary.

    temp = random_complex_tensor(chi_lower, 4)
    U, S, V = tensorsvd(temp, [1,2], [3,4])
    u = ncon((U, V), ([-1,-2,1], [1,-3,-4]))

    temp = random_complex_tensor(chi_lower, 4)
    U, S, V = tensorsvd(temp, [1,2], [3,4])
    w = ncon((U, V), ([-1,-2,1], [1,-3,-4]))
    w = reshape(w, (chi_lower^2, chi_lower, chi_lower))
    # Truncate to the first chi_upper singular values
    w = w[1:chi_upper,:,:]

    udag = permutedims(conj(u), (3,4,1,2))
    wdag = permutedims(conj(w), (2,3,1))
    return Layer(Disentangler(u), Isometry(w))
end

function dm4pureState(pureState)
    dm = ncon((conj(pureState),pureState),([-100,-200,-300],[-400,-500,-600])) |> complex
    return dm
end

# Maybe one can improve on the Power method by the Lanczos method?
function fixedpoint(Sop; seed_state=threesiteeye(chi), loop::Int64=10)
    # Sop is the operator whose fixed-point we seek

    # Should we do this by naive Power method on seed_state
    state = seed_state
    for i in 1:loop
        state = Sop(state)
    end

    return state

    # or use some routine from tensoreig?
        # E,U = tensoreig(Sop, [1,2,3], [4,5,6], hermitian=true)
        # newTop = U[:,:,:,1]
        # threeSiteEnergy = E[1]
        # return newTop, threeSiteEnergy
        # How is it that this could possibly return the energy?
        # Only because the levelTensors have been optimized for a particular Hamiltonian :-?
end

# cFuncs = Map(Asop/Dsop,layerList)
# with the possible need of a reverse if one starts from the other end of the MERA
#function buildOpList(op,cFuncs::list_of_compose_fns)
#end

# Generalize this to buildOpList(op, Array{Asop/Dsop(LevelTensors)})
function buildReverseRhosList(m::MERA, top_n=length(m.levelTensors))
    # Specify the number of EvalScales sought. If not provided, defaults to all EvalScales
    # evalscale starts at zero below layer1
    uw_list=m.levelTensors;
    totLayers = length(uw_list)
    stateAtEvalScale = getTopState(m.topLayer)
    rhosListReverse = [];
    push!(rhosListReverse,m.topLayer.state)
    push!(rhosListReverse,stateAtEvalScale)

    for j in reverse((totLayers-top_n+1):totLayers)
        stateAtEvalScale = descend_threesite_symm(stateAtEvalScale,uw_list[j])
        push!(rhosListReverse,stateAtEvalScale)
    end

    # Returns state above TopLayer and then the next n_top states
    # To get the state at later x, with zero at the topmost layer, access rhosListReverse[1+x]
    return rhosListReverse
end

# Note that this function does an expensive copy() of all the levelTensors in a MERA
# For that reason it is not ideal for use inside buildReverseRhosList() or buildOpList()
# if those will be called repeatedly. Appending a layer would be easy if the layers were
# represented as a linked list, or a lookup table with pointers to the actual levelTensors
# The tradeoff would be that one would have to do a small number of hops to "randomly" start
# at some layer. Interestingly, this would have been easy if this list was constructed lazily
# by the consumer rather than eagerly by the producer.
function getLayerList(m::MERA;topRepeat::Int64=1)
    local layers::Array{Layer,1}
    layers = copy(m.levelTensors)
    if(topRepeat>0)
        append!(layers, fill(m.topLayer.levelTensors,topRepeat))
    end
    return layers
end

function hamSpectrumLayerwise(h_base,layers::Array{Layer,1};downshift::Float=0.0)
    function sqreshape(op)
        l = length(op) |> sqrt |> Int
        return reshape(op,(l,l))
    end

    function findspectrum(op;spectrum_downshift::Float=0.0)
        D, V = eig(Hermitian(   0.5*(sqreshape(op)+sqreshape(conj(op))  )  ))
        return (D .+ spectrum_downshift) ./ 3
        # Since we're feeding in a three-site operator
    end

    spectrum_0 = findspectrum(h_base;spectrum_downshift=downshift)
    spectrum_list = [spectrum_0]
    h_layer = h_base
    for i in 1:length(layers)
        # Step the Hamiltonian up for the next iteration and find its spectrum
        h_layer = ascend_threesite_symm(h_layer, layers[i])
        spectrum_i = h_layer |> (x -> findspectrum(x;spectrum_downshift=downshift))
        # Push into container
        spectrum_list = [spectrum_list...,spectrum_i]
        # Model as Array{Any,1} (currently used) or as tuple of tuples?
        # or use DataArray or DataFrame
    end

    #spectrum_end = ascend_threesite_symm(h_layer, m.topLayer.levelTensors) |> findspectrum
    #spectrum_list = [spectrum_list...,spectrum_end]

    # Return list of lists
    return spectrum_list
end

function imposePDBC(op::LocalOperator)
    return (ncon((op), ([-100,-200,-300,-400,-500,-600]))
    				+ ncon((op), ([-300,-100,-200,-600,-400,-500]))
    				+ ncon((op), ([-200,-300,-100,-500,-600,-400])) )
    # Divide by 3 here IFF the 3site Hamiltonian doesn't already account for that
end

#end
