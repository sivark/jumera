# ------------------------------------------------------------
# [TODO] Autogenerate environments
# 1. Understand why these are the optimal contraction orderings for each case
# 2. Can I automate the creation of these different environments? Either with a function or with a macro
# 3. For each tensor, what is the environment? Find and evaluate
# 4. Write a macro to go through the fully contracted network,
#       and knock off an isometry and label the open legs [-100,-200,-300,-400]
# ------------------------------------------------------------

using JLD

function improveU(h_layer::Array{Complex{Float64},6}, l::Layer, rho_layer::Array{Complex{Float64},6}, params::Dict)
    u = l.u.elem
    u_dg = l.udag.elem
    w = l.w.elem
    w_dg = l.wdag.elem
    h = h_layer
    rho = rho_layer

    for i in 1:params[:Qsingle]
        env1 = ncon((rho,
                     w, w, w,
                     u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([17,18,10,15,14,9],
                     [15,5,6], [14,16,-1], [9,-2,8],
                     [6,16,1,2],
                     [1,2,-3,3,4,13],
                     [3,4,7,12], [13,-4,11,19],
                     [5,7,17], [12,11,18], [19,8,10]))

        env2 = ncon((rho,
                     w, w, w,
                     u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([4,15,6,3,10,5],
                     [3,1,11], [10,9,-1], [5,-2,2],
                     [11,9,12,19],
                     [19,-3,-4,18,7,8],
                     [12,18,13,14], [7,8,16,17],
                     [1,13,4], [14,16,15], [17,2,6]))

        env3 = ncon((rho,
                     w, w, w,
                     u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([6,15,4,5,10,3],
                     [5,2,-1], [10,-2,9], [3,11,1],
                     [9,11,19,12],
                     [-3,-4,19,8,7,18],
                     [8,7,17,16], [18,12,14,13],
                     [2,17,6], [16,14,15], [13,1,4]))

        env4 = ncon((rho,
                     w, w, w,
                     u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([10,18,17,9,14,15],
                     [9,8,-1], [14,-2,16], [15,6,5],
                     [16,6,2,1],
                     [-4,2,1,13,4,3],
                     [-3,13,19,11], [4,3,12,7],
                     [8,19,10], [11,12,18], [7,5,17]))

        envTot = env1 + env2 + env3 + env4

        # u is a map from [3,4] to [1,2].
        # The environment must be a map from [1,2] (Bdag) to [3,4] (A) so that we can trace the product
        # But the convention for the environment here is transposed! Therefore we need not transpose answer for U!
        # Minus sign not too important because we're skipping for both U and Udagger, so it becomes a matter of convention?
        #Bdag,S,A = tensorsvd(envTot, [1,2] , [3,4])
        #improved_u = permutedims( conj(ncon(Bdag,A),([-1,-2,1],[1,-3,-4]))) ,([-1,-2,1],[1,-3,-4])), (3,4,1,2) )

        # Note that V is daggered already, wrt usual SVD convention
        U,S,V = tensorsvd(envTot, [1,2] , [3,4])
        improved_u = (-1)*ncon((conj(U),conj(V)),([-1,-2,1],[1,-3,-4]))
        u = improved_u
        u_dg = permutedims(conj(u), (3,4,1,2))
    end
    return u
end

function improveW(h_layer::Array{Complex{Float64},6}, l::Layer, rho_layer::Array{Complex{Float64},6}, params::Dict)
    u = l.u.elem
    u_dg = l.udag.elem
    w = l.w.elem
    w_dg = l.wdag.elem
    h = h_layer
    rho = rho_layer

    for i in 1:params[:Qsingle]
        env1 = ncon((rho,
                     w, w,
                     u, u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([16,15,19,18,17,-1],
                     [18,5,6], [17,9,8],
                     [6,9,2,1], [8,-2,10,11],
                     [2,1,10,4,3,12],
                     [4,3,7,14], [12,11,13,20],
                     [5,7,16], [14,13,15], [20,-3,19]))

        env2 = ncon((rho,
                     w, w,
                     u, u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([18,17,19,16,15,-1],
                     [16,12,13], [15,5,6],
                     [13,5,9,7], [6,-2,2,1],
                     [7,2,1,8,4,3],
                     [9,8,14,11], [4,3,10,20],
                     [12,14,18], [11,10,17], [20,-3,19]))

        env3 = ncon((rho,
                     w, w,
                     u, u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([19,20,15,18,-1,14],
                     [18,5,6], [14,17,13],
                     [6,-2,2,1], [-3,17,12,11],
                     [2,1,12,4,3,9],
                     [4,3,7,10], [9,11,8,16],
                     [5,7,19], [10,8,20], [16,13,15]))

        env4 = ncon((rho,
                     w, w,
                     u, u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([15,20,19,14,-1,18],
                     [14,13,17], [18,6,5],
                     [17,-2,11,12], [-3,6,1,2],
                     [12,1,2,9,3,4],
                     [11,9,16,8], [3,4,10,7],
                     [13,16,15], [8,10,20], [7,5,19]))

        env5 = ncon((rho,
                     w, w,
                     u, u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([19,17,18,-1,15,16],
                     [15,6,5], [16,13,12],
                     [-3,6,1,2], [5,13,7,9],
                     [1,2,7,3,4,8],
                     [3,4,20,10], [8,9,11,14],
                     [-2,20,19], [10,11,17], [14,12,18]))

        env6 = ncon((rho,
                     w, w,
                     u, u,
                     h,
                     u_dg, u_dg,
                     w_dg, w_dg, w_dg),
                    ([19,15,16,-1,17,18],
                     [17,8,9], [18,6,5],
                     [-3,8,11,10], [9,6,1,2],
                     [10,1,2,12,3,4],
                     [11,12,20,13], [3,4,14,7],
                     [-2,20,19], [13,14,15], [7,5,16]))

        envTot = env1 + env2 + env3 + env4 + env5 + env6
        U,S,V = tensorsvd(envTot, [1], [2,3])
        improved_w = (-1)*ncon((conj(U), conj(V)), ([-1,1], [1,-2,-3]))
        w = improved_w
        w_dg = permutedims(conj(w), (2,3,1))
    end
    return w
end

function improveLayer(h_layer::Array{Complex{Float64},6}, l::Layer, rho_layer::Array{Complex{Float64},6}, params::Dict)
    for i in 1:params[:Qlayer]
        #println(size(h_layer),"improvelayer",i)
        u = improveU(h_layer, l, rho_layer, params)
        w = improveW(h_layer, l, rho_layer, params)
        l = Layer(Disentangler(u),Isometry(w))
    end
    return l
end

function improveTop(h_layer::Array{Complex{Float64},6}, m::MERA)
    # Imposing periodic BCs
    h_pdBC = (ncon((h_layer), ([-100,-200,-300,-400,-500,-600]))
				+ ncon((h_layer), ([-300,-100,-200,-600,-400,-500]))
				+ ncon((h_layer), ([-200,-300,-100,-500,-600,-400])) )/3
    # pulling out the lowest energy eigenvector?
    E,U = tensoreig(h_pdBC, [1,2,3], [4,5,6], hermitian=true)
    newTop = U[:,:,:,1]
    threeSiteEnergy = E[1]
    return newTop, threeSiteEnergy
    #println(energy,"improveTop")
end

function buildReverseRhosList(m::MERA, top_n=length(m.levelTensors))
    # Specify the number of EvalScales sought. If not provided, defaults to all EvalScales
    # evalscale starts at zero below layer1
    uw_list=m.levelTensors;
    totLayers = length(uw_list)
    stateAtEvalScale = ncon((conj(m.topTensor),m.topTensor),([-100,-200,-300],[-400,-500,-600])) |> complex
    rhosListReverse = [];
    push!(rhosListReverse,stateAtEvalScale)
    for j in reverse((totLayers-top_n+1):totLayers)
        stateAtEvalScale = descend_threesite_symm(stateAtEvalScale,uw_list[j])
        push!(rhosListReverse,stateAtEvalScale)
    end
    return rhosListReverse
end

# Write a function to train the n coarsest layers, and also the top tensor
# By default, the number of layers is the whole MERA
function improveGraft!(h_base::Array{Complex{Float64},6}, m::MERA, params::Dict, top_n=length(m.levelTensors))
    #uw_list = m.levelTensors
    H = reshape(h_base, (8*8*8,8*8*8))
    D, V = eig(Hermitian(H))
    D_max = D[end]
    energyPerSite = 0.0

    len = length(m.levelTensors)
    h_layer = ascendTo(h_base, m, (length(m.levelTensors)-top_n) )
    #println(size(h_layer))

    # we need the state only at levels coarser than the ones we're training
    rhoslist_partial_rev = buildReverseRhosList(m, top_n-1)
    rhoslist_snapshots = []

    # [TODO] Convert this to a while with a check on :EnergyDelta also
    for i in 1:params[:Qsweep]
        h_layer = ascendTo(h_base, m, (length(m.levelTensors)-top_n) )
        for j in collect(len-top_n+1:len)
            m.levelTensors[j] = improveLayer(h_layer, m.levelTensors[j], rhoslist_partial_rev[len-j+1], params)
            h_layer = ascend_threesite_symm(h_layer,m.levelTensors[j])
            #println(size(h_layer),"improvegraft")
        end
        m.topTensor, threeSiteEnergy =  improveTop(h_layer, m)
        #println(threeSiteEnergy,"improveGraft!")
        energyPerSite = (threeSiteEnergy + Dmax)/3

        if(i%50 == 1)
            println(i, ":", energyPerSite)
            push!(rhoslist_snapshots,   (buildReverseRhosList(m) |> reverse)   )
            # Pushes the full rhoslist instead of just the top few layers we used
        end
    end
    save("rhoslist_snapshots_$(length(m.levelTensors))layers.jld", "rhoslist_snapshots_$(top_n)smoothing", rhoslist_snapshots)
    println(string(map((x) -> '-', collect(1:28))...))
    return energyPerSite
end
