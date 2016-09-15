using JLD

function readRhosList(len::Int,n_top)
    rhoslist_list = load("rhoslist_snapshots_$(len)layers.jld", "rhoslist_snapshots_$(n_top)smoothing")
    return rhoslist_list
end

function getEntList(rlist)
    Slist::Array{Float64}
    Slist=map(entropy,rlist)
    return Slist
end

function entropy(rho3site)
    EE::Float64
    U,S,Vdag=tensorsvd(rho3site, [1,2,3], [4,5,6])
    EE = mapreduce( (x) -> -x*log2(x) , +, S)
    return EE
end

