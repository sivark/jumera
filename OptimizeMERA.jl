#----------------------------------------------------------------------------------------------------
# [Bikeshedding] Can this code be made independent on the MERA scheme? (2-> MERA with 3-site operators)
#   -   In principle, I can change 6 index objects to any even number,
#       and if there isn't a match between the state and an operator then we'll see errors from ncon()
#   -   OptimizeMERA module will then have to import the appropriate MERA structure module.
# [Stretch TODO] Autogenerate environments
#   -   Can I automate the creation of these different environments? Either with a function or with a macro
#   -   For each tensor, what is the environment? Find and evaluate
#   -   Write a macro to go through the fully contracted network,
#       and knock off an isometry and label the open legs [-100,-200,-300,-400]
#----------------------------------------------------------------------------------------------------

using JLD

function improveU(h_layer::LocalOperator, l::Layer, rho_layer::LocalOperator, params::Dict)
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

function improveW(h_layer::LocalOperator, l::Layer, rho_layer::LocalOperator, params::Dict)
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

function improveLayer(h_layer::LocalOperator, l::Layer, rho_layer::LocalOperator, params::Dict)
    for i in 1:params[:Qlayer]
        #println(size(h_layer),"improvelayer",i)
        u = improveU(h_layer, l, rho_layer, params)
        w = improveW(h_layer, l, rho_layer, params)
        l = Layer(Disentangler(u),Isometry(w))
    end
    return l
end


"""
# Write a function to train the n coarsest layers, and also the top tensor
# By default, the number of layers is the whole MERA; and non scale invariant top layer?
# *Make this a HOfunction that takes in a function for improving the top layer!*
# Would be wonderful if functions have useful type information! :P


#----------------------------------------------------------------------------------------------------
#   We have to make a choice between
#   1. Building the list of effective Hamiltonians for each layer and then recursively
#       stepping down the optimized state through optimized tensors.
#   2. Building the list of states at each layer and then progressively stepping up the Hamiltonian
#        after optimizing tensors on a layer. [CURRENTLY USING THIS]
#
#   I don't see any inherent advantage to one over the other. Both Asop & Dsop are 3site-->3site, so
#   both must have the same computational cost.
#   but one of them might be stable and the other unstable :-? Maybe I should either:
#
#   1. Make two functions for those two options and pass the choice explicitly to improveGraft!().
#   2. make one function (since Rho and Ham are functionally interchangeable) and choose one of the
#       two modes when calling the function.
#
#   It seems like I could just swap (UV,Ham)<-->(IR,Rho) in this code, and things will still work!?
#   -------------------------------------------------------------------------------------------------
"""
function improveGraft!(improveTopLayer::Function,h_base::LocalOperator, m::MERA, params::Dict, top_n=length(m.levelTensors))
    #uw_list = m.levelTensors
    H = reshape(h_base, (8*8*8,8*8*8))
    D, V = eig(Hermitian(H))
    D_max = D[end]

    len = length(m.levelTensors)
    h_layer = ascendTo(h_base, m, (length(m.levelTensors)-top_n) )
    #println(size(h_layer))

    # we need the state only at levels coarser than the ones we're training
    rhoslist_partial_rev = buildReverseRhosList(m, top_n-1)
    rhoslist_snapshots   = []
    #Array(Array(LocalOperator,len),:Qsweep)

    fractional_energy_change    = convert(Float,1.0);
    energyPerSiteOld            = convert(Float,0.0);
    energyPerSite               = convert(Float,0.0);
    i = 1;
    while(!stopCondition(len, i, fractional_energy_error(energyPerSite, len), fractional_energy_change, params))
        for b in 1:params[:Qbatch]
            # Ascend Hamiltonian to the layer we want to optimize
            h_layer = ascendTo(h_base, m, (length(m.levelTensors)-top_n) )

            # Starting from the UV because that is where we have useful information about the state (the Hamiltonian)
            # Move to the IR, progressively optimizing the layers we care about
            for j in collect(len-top_n+1:len)
                m.levelTensors[j] = improveLayer(h_layer, m.levelTensors[j], rhoslist_partial_rev[len-j+2], params)
                h_layer = ascend_threesite_symm(h_layer,m.levelTensors[j])
                #println(size(h_layer),"improvegraft")
            end
            m.topLayer, threeSiteEnergy =  improveTopLayer(h_layer, m.topLayer, params)
            energyPerSite = (threeSiteEnergy + Dmax)/3
            # [IMPORTANT] **Use this only if the randomly initialized layer is inserted into m.levelTensors**
            # Crazy if we wreck previously optimized m.topLayer.levelTensors with randomly initialized (new) toplayer
            # Whether we push the trained topLayer into m.levelTensors or not, we must start by optimizing the
            # randomly initialized layer instead of wrecking the trained layer.

            # Generate new RhosList for next round of optimization
            rhoslist_partial_rev = buildReverseRhosList(m, top_n-1)
        end

        # Computing for each BATCH rather than for each ITERATION
        fractional_energy_change = ((energyPerSite - energyPerSiteOld)/energyPerSite) |> abs;
        energyPerSiteOld = energyPerSite;

        #print status at the end of every BATCH
        @printf "%4d iter: E = %1.11f , rate of change = %1.1e , fractional error = %1.1e\n" i*params[:Qbatch] energyPerSite fractional_energy_change fractional_energy_error(energyPerSite, len)
        push!(rhoslist_snapshots,   (buildReverseRhosList(m) |> reverse)   )
        # Pushes the full rhoslist instead of just the top few layers we used

        i+=1;
    end

    println(string(map((x) -> '-', collect(1:28))...))
    return rhoslist_snapshots
    # Does removing the return value affect the speed of the program?
end

function stopCondition(nLyr::Int64, sweepCounter::Int64, fractional_energy_error::Float, fractional_energy_change::Float, params::Dict)
    # Higher layers contribute less to the energy (and its improvement), still persist with
    # training them to get the correct entanglement structure
    return ( (fractional_energy_error / fractional_energy_change) > (params[:GiveUp]*nLyr^2)  || sweepCounter>params[:Qsweep])
    #(i<=params[:Qsweep] || fractional_energy_change>params[:EnergyDelta])
    #(i<=params[:Qsweep] && fractional_energy_change>params[:EnergyDelta])
end

function improveNonSILtop(h_below::LocalOperator, t::TopLayer, params::Dict)
    local newTopLayer::TopLayer
    local threeSiteEnergy::Float
    threeSiteEnergy = 0.0
    rhoTop = t.state
    newLevelTensors = t.levelTensors
    newTopState = t.state
    for i in 1:params[:Qtop]
        newLevelTensors = improveLayer(h_below, t.levelTensors, rhoTop, params)

        h_above = ascend_threesite_symm(h_below, t.levelTensors)

        # Imposing periodic BCs
        h_pdBC  = imposePDBC(h_above)

        # pulling out the lowest energy eigenvector
        E,U = tensoreig(h_pdBC, [1,2,3], [4,5,6], hermitian=true)
        newTopTensor = U[:,:,:,1]
        newTopState = dm4pureState(newTopTensor)
        threeSiteEnergy = E[1]
    end

    newTopLayer = TopLayer(newLevelTensors,newTopState)

    return newTopLayer, threeSiteEnergy
end

function improveSILtop(h_below::LocalOperator, t::TopLayer, params::Dict)
    local newTopLayer::TopLayer
    local threeSiteEnergy::Float
    # Get layer tensors, hamiltonian at the start of layer
    sil = t.levelTensors
    rho_top = t.state

    for ctr in 1:params[:Qtop]
        # Resum hamiltonian to include number of layers we'd like to keep track of, in geometric weight
        # Idea being that we actually need infinity, but were truncating for practicality
        function resum(h,n_resum::Int64=3)
            h_resummed = h;
            # In principle, we must construct a list of h_level and then evaluate on each level.
            # But assuming scale invariance and linear operation of state_layer on h_layer we can factor out state_layer
            for r in 1:n_resum
                h_resummed = h + (0.5)*ascend_threesite_symm(h_resummed,sil)
            end
            h_resummed = h_resummed / (n_resum + 1)
            #resum=0 gives the usual result
            return h_resummed
        end

        # Improve layer with this resummed hamiltonian
        sil = improveLayer(     resum(h_below, 3) , sil, rho_top, params)

        # Construct state to be the fixed-point of the descending superoperator
        # Plausibly, the finite-size topTensor we've found has some overlap with the thermodynamic ground state
        # Instead of supplying this, should we supply the identity instead?
        rho_top = fixedpoint(    Dsop(sil) , seed_state=rho_top  );
    end

    # NCon returns a complex number; we expect the energy to be real
    threeSiteEnergy = real(expectation(  ascend_threesite_symm(h_below, sil), rho_top))
    newTopLayer = TopLayer(sil,rho_top)

    return newTopLayer, threeSiteEnergy
end

function growMERA!(m::MERA,LAYER_SHAPE,INIT_LAYERS)
    for lyr in (INIT_LAYERS+1):(length(LAYER_SHAPE)-1)
        @printf "\nNow adding layer number: %2d of bond dimensions %2d -> %2d\n" lyr LAYER_SHAPE[lyr] LAYER_SHAPE[lyr+1]
        exact_persite = exact_energy_persite(lyr);
        @printf "Not always exact per-site energy for this depth: %1.11f\n" exact_persite

        # Ensure that newly initialized layer gets trained first!
        newLayer = generate_random_layer(LAYER_SHAPE[lyr],LAYER_SHAPE[lyr+1])
        push!(m.levelTensors, newLayer)
        #push!(m.levelTensors, m.topLayer.levelTensors)
        #m.topLayer = generate_random_top(LAYER_SHAPE[lyr],LAYER_SHAPE[lyr+1])

        # It is tempting to guess a better initialization for the new layer.
        # Ideally, it is tempting to use the penultimate layer,
        # since they must be the same at the critical point, etc...
        # but local "gauge" freedom will probably make that quite useless

        jldopen("rhoslist_snapshots_$(length(m.levelTensors))layers.jld","w") do file
            # Improving the newly added layer and the top tensor
            rhoslist_snapshots1 = improveGraft!(improveNonSILtop,isingH, m, parameters_graft, 0)
            # It's important to iterate over the top layer several times,
            # otherwise it will wreck the lower layers when we sweep!
            write(file, "rhoslist_snapshots_1smoothing", rhoslist_snapshots1)

            rhoslist_snapshots2 = improveGraft!(improveNonSILtop,isingH, m, parameters_sweep, 1)
            write(file, "rhoslist_snapshots_2smoothing", rhoslist_snapshots2)

            #rhoslist_snapshots3 = improveGraft!(improveNonSILtop,isingH, m, parameters_shortsweep, 2)
            #write(file, "rhoslist_snapshots_3smoothing", rhoslist_snapshots3)

            # sweep over all layers
            rhoslist_snapshotsAll = improveGraft!(improveNonSILtop,isingH, m, parameters_sweep)
            write(file, "rhoslist_snapshots_$(lyr)smoothing", rhoslist_snapshotsAll)
        end

        # println("\nFinal energy of this optimized MERA: ", energy_persite)
        # println("Not always exact per-site energy for this depth: ", exact_persite,"\n")
        # println("Fractional error in our variational estimate: ", (energy_persite - exact_persite)/(exact_persite) )
        save("solutionMERA_$(lyr)layers_$(LAYER_SHAPE[1:lyr+1])shape.jld", "m_$(lyr)layers", m)
        println(string(map((x) -> '-', collect(1:28))...))
        println()
    end
end

# DO NOT USE FOR NOW
# function floatlenprint()
#     # Assuming fprintf() convention???
#     if (Float == Float32)
#         return ASCIIString("f")
#     elseif (Float == Float64)
#         return ASCIIString("lf")
#     end
# end
