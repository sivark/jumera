# ------------------------------------------------------------
# 1. Understand why these are the optimal contraction orderings for each case
# 2. Can I automate the creation of these different environments? Either with a function or with a macro
# 3. For each tensor, what is the environment? Find and evaluate
# 4.  Write a macro to go through the fully contracted network,
#       and knock off an isometry and label the open legs [-100,-200,-300,-400]
# 5. SVD the environment, find UVdag and update the tensor!
# ------------------------------------------------------------



function improveU(h_layer, l::Layer, rho_layer)
    u = l.u.elem
    u_dg = l.udag.elem
    w = l.w.elem
    w_dg = l.wdag.elem
    h = h_layer
    rho = rho_layer


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

    U,S,V = tensorsvd(envTot, [1,2] , [3,4])
    improved_u = ncon((conj(U),conj(V)),([-1,-2,1],[1,-3,-4]))

    return improved_u
end


function improveW(h_layer, l::Layer, rho_layer)
    u = l.u.elem
    u_dg = l.udag.elem
    w = l.w.elem
    w_dg = l.wdag.elem
    h = h_layer
    rho = rho_layer

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
    improved_w = ncon((conj(U), conj(V)), ([-1,1], [1,-2,-3]))
    return improved_w
end

function improveLayer(h_layer, l::Layer, rho_layer, params)
    for i in 1:params[:layerIters]
        u = improveU(h_layer, l, rho_layer)
        w = improveW(h_layer, l, rho_layer)
        l = Layer(Disentangler(u),Isometry(w))
        return l
    end
end

function improveTop(h_layer, m::MERA)
#     U,S,V = tensorsvd(h_layer,[1,2,3],[4,5,6])
#     # since h is hermitean, U = V-dag
#     println(size(U))
#     println(typeof(U))
#     return U

    # Imposing periodic BCs
    h = h_layer

    # pulling out the lowest energy eigenvector?
    newTop = tensoreig(h_layer, [1,2,3], [4,5,6], hermitian=true)[2][:,:,:,1]
    return newTop
end

function buildrhoslist(m::MERA)
    # evalscale starts at zero below layer1
    uw_list=m.levelTensors;
    totLayers = length(uw_list)

    stateAtEvalScale = ncon((conj(m.topTensor),m.topTensor),([-100,-200,-300],[-400,-500,-600])) |> complex

    rholistReverse = [];

    for j in reverse(1:totLayers)
        #println(j)
        stateAtEvalScale = descend_threesite_symm(stateAtEvalScale,uw_list[j])
        push!(rholistReverse,stateAtEvalScale)

    end

    #println(size(rholistReverse))

    rholist = reverse(rholistReverse)
    return rholist
end

function improveMERA!(m::MERA, h, params)
    h_orig = complex(h)

    energy = complex(0.0)
    energyChangeFraction = 1.0

    counter = 0

    while( energyChangeFraction > params[:energyDelta] && counter < params[:maxIter])
        h_layer = h_orig

        # pre-build rhos at every layer
        # since we cannot iterative descend less
        # and this does not need updated layer
        rholist = buildrhoslist(m);

        newlayerList = [];

        counter += 1
        oldEnergy = energy

        for (l,rho_layer) in zip(m.levelTensors,rholist)
            l = improveLayer(h_layer, l, rho_layer, params)

            push!(newlayerList,l)
            # ascend h by one layer
            h_layer = ascend_threesite_symm(h_layer,l)
        end

        # handle the top
        newTop = improveTop(h_layer,m)

        # ACTUALLY MODIFY the network
        # m = MERA(newlayerList,newTop)
        m.levelTensors = newlayerList;
        m.topTensor = newTop;

        # compute the energy now, at any evalscale of your choice
        # in practice ascending might be cheaper than descending?

        energy = expectation(h_orig,m,0)  |>  (x)->reshape(x,1)[1]

        energyChangeFraction = abs( (energy - oldEnergy) / energy )

        println(counter," -- ",energy," -- ",energyChangeFraction)
    end
    #return m
end
