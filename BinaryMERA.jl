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

immutable Layer
    # first disentangle and then coarsegrain
    # so the bond-dimension given out by U must match the bond dimension taken in by W
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

    # Also, can we enforce the "right" network connection where
    # the V inputs come from one U outputs and all U outputs go to V inputs?
end

#MUTABLE, in case we want to optimize in-place?
# To save on memory, we can make this be a container class?
# Or should we make this immutable and expect Julia to run GC
# every time we create a new MERA object and the old one aint needed?

type MERA
    levelTensors::Array{Layer} # sequence of layers
    topTensor::Array{Complex{Float},3} # 3 indices
end

# Does it make sense to specify the bond-dimension as an input to each of these types,
# so that one can sanity-check the disentangler and isometry?
# Not quite straightforward, since we might want a MERA where the bond dimension changes with layer :-?

# ------------------------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------------------------

function random_complex_tensor(chi, rank)
    res::Array{Complex{Float},rank}
    real = randn(ntuple(_ -> chi, rank)...)
    imag = randn(ntuple(_ -> chi, rank)...)
    res = real + im*imag
    return res
end

function generate_random_layer(chi_lower,chi_upper)
    # Generate a random tensor and SVD it to get a random unitary.

    temp = random_complex_tensor(chi_lower, 4)
    U, S, V = tensorsvd(temp, [1,2], [3,4])
    u = ncon((U, V), ([-1,-2,1], [1,-3,-4]))

    temp = random_complex_tensor(chi_lower, 4)
    U, S, V = tensorsvd(temp, [1,2], [3,4])
    w = ncon((U, V), ([-1,-2,1], [1,-3,-4]))
    w = reshape(w, (chi_lower^2, chi_lower, chi_lower))
    # Truncate to the first chi singular values
    w = w[1:chi_upper,:,:]

    udag = permutedims(conj(u), (3,4,1,2))
    wdag = permutedims(conj(w), (2,3,1))
    return Layer(Disentangler(u), Isometry(w))
end
# function generate_random_layer(chi)
#     # Generate a random tensor and SVD it to get a random unitary.
#
#     temp = random_complex_tensor(chi, 4)
#     U, S, V = tensorsvd(temp, [1,2], [3,4])
#     u = ncon((U, V), ([-1,-2,1], [1,-3,-4]))
#
#     temp = random_complex_tensor(chi, 4)
#     U, S, V = tensorsvd(temp, [1,2], [3,4])
#     w = ncon((U, V), ([-1,-2,1], [1,-3,-4]))
#     w = reshape(w, (chi^2, chi, chi))
#     # Truncate to the first chi singular values
#     w = w[1:chi,:,:]
#
#     udag = permutedims(conj(u), (3,4,1,2))
#     wdag = permutedims(conj(w), (2,3,1))
#
# #     # VERIFYING that
# #     # u*udag and udag*u give 1
# #     # w*wdag gives 1 and wdag*w does not
#
# #     foo1=ncon((u,udag),([-100,-200,1,2],[1,2,-300,-400]))
# #     foo2=ncon((udag,u),([-100,-200,1,2],[1,2,-300,-400]))
# #     foo1=reshape(foo1,(chi^2,chi^2))
# #     foo2=reshape(foo2,(chi^2,chi^2))
# #     println(vecnorm(foo1-eye(chi^2)))
# #     println(vecnorm(foo2-eye(chi^2)))
#
# #     foo3=ncon((w,wdag),([-100,1,2],[1,2,-200]))
# #     foo4=ncon((wdag,w),([-100,-200,1],[1,-300,-400]))
# #     foo4=reshape(foo4,(chi^2,chi^2))
# #     println(vecnorm(foo3-eye(chi)))
# #     println(ncon((foo4),([1,1])))
#
#     return Layer(Disentangler(u), Isometry(w))
# end



function generate_random_top(chi)
    top = randn(ntuple(_ -> chi, 3)...)
    top /= vecnorm(top)
    return top
end

function generate_random_MERA(listOfChis)
    uw_list = []
    for i in 1:(length(listOfChis)-1)
        push!(uw_list, generate_random_layer(listOfChis[i],listOfChis[i+1]) )
    end
    topTensor = generate_random_top(listOfChis[end])

    return MERA(uw_list, topTensor)
end
# function generate_random_MERA(chi,n_layers)
#     uw_list = []
#     for i in 1:n_layers
#         push!(uw_list, generate_random_layer(chi) )
#     end
#     topTensor = generate_random_top(chi)
#
#     return MERA(uw_list, topTensor)
# end

# ------------------------------------------------------------
# BUILDING SUPER-OPERATORS
# Note that the rightside operator is the parity flip rotation of the leftside operator :-)
# ------------------------------------------------------------

function ascend_threesite_left(op::Array{Complex{Float},3*2}, l::Layer)
    scaled_op::Array{Complex{Float},3*2}
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

function ascend_threesite_right(op::Array{Complex{Float},3*2}, l::Layer)
    scaled_op::Array{Complex{Float},3*2}
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

function ascend_threesite_symm(op::Array{Complex{Float},3*2}, l::Layer)
    return 0.5*( ascend_threesite_left(op,l)+ascend_threesite_right(op,l) )
end


function descend_threesite_right(op::Array{Complex{Float},3*2}, l::Layer)
    scaled_op::Array{Complex{Float},3*2}
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

function descend_threesite_left(op::Array{Complex{Float},3*2}, l::Layer)
    scaled_op::Array{Complex{Float},3*2}
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

function descend_threesite_symm(op::Array{Complex{Float},3*2}, l::Layer)
    return 0.5*( descend_threesite_left(op,l)+descend_threesite_right(op,l) )
end

function ascendTo(op::Array{Complex{Float},3*2},m::MERA,EvalScale::Int64)
    opAtEvalScale = op
    for i in collect(1:EvalScale)
        opAtEvalScale = ascend_threesite_symm(opAtEvalScale,m.levelTensors[i])
    end
    return opAtEvalScale
end

function descendTo(m::MERA,EvalScale::Int)
    # evalscale starts at zero below layer1
    totLayers = length(m.levelTensors)
    stateAtEvalScale = ncon((conj(m.topTensor),m.topTensor),([-100,-200,-300],[-400,-500,-600])) |> complex
    for j in reverse((EvalScale+1):totLayers)
        stateAtEvalScale = descend_threesite_symm(stateAtEvalScale,m.levelTensors[j])
    end
    return stateAtEvalScale
end

function expectation(op,rho)
    # Need operator and rho to be given at the same scale
    # Scale for operators is 1-indexed while
    # Scale for states is 0-indexed at the ultraviolet cutoff
    result = ncon((op,rho),([1,2,3,4,5,6],[4,5,6,1,2,3]))
    return result
end
