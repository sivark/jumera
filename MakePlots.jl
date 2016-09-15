using JLD

function readRhosList(len::Int,n_top)
    rhoslist_list = load("rhoslist_snapshots_$(len)layers.jld", "rhoslist_snapshots_$(n_top)smoothing")
    return rhoslist_list
end

