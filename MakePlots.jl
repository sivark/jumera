#module examineMERA

using JLD
using Plots
pyplot()
#gr()


# Put test and profiling also in this module?

function readRhosList(len::Int,n_top)
    rhoslist_list = load("rhoslist_snapshots_$(len)layers.jld", "rhoslist_snapshots_$(n_top)smoothing")
    return rhoslist_list
end

function getEntList(rlist)
    local Slist::Array{Float64}
    Slist=map(entropy,rlist)
    return Slist
end

function entropy(rho3site)
    local EE::Float64
    U,S,Vdag=tensorsvd(rho3site, [1,2,3], [4,5,6])
    EE = mapreduce( (x) -> -x*log2(x) , +, S)
    return EE
end


function animatedplot(filename::String;n_start::Int=1,n_stop::Int=0,n_smoothing_list=())
    p=plot([],[],legend=false,
            xaxis=("Layer",(0,11),0:1:10),
            yaxis=("Entanglement",(0,8),0:1:8),
            #background_color=RGB(0.3,0.5,0.1)
            title=("Entanglement -vs- layer");
            );
    #title!("Entanglement -vs- layer");
    hline!([1.6],line=(0.5,:dash,0.6,:red))
    anim = Animation()
    for n_layers in collect(n_start:n_stop)
        # Evolving arrangement of MERA with current no. layers
        # Add our set of n_top to the tuple which is being scanned
        for n_smoothing in (n_smoothing_list...,n_layers)
            for elist in map(getEntList, readRhosList(n_layers,n_smoothing) )
                plot!(collect(0:(length(elist)-1)), elist)
                frame(anim)
            end
        end
        # Final arrangement of MERA with this or fewer layers
        empty!(p.series_list)
        hline!([1.6],line=(0.5,:dash,0.6,:red))
        for j in collect(n_start:n_layers)
            elist = readRhosList(j,j)[end] |> getEntList
            plot!(collect(0:(length(elist)-1)), elist,
            line=(0.4,:darkblue))
        end
        frame(anim)
    end
    return gif(anim,filename,fps=4,loop=1)
end

