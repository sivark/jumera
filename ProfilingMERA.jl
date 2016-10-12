typealias Float Float64

include("BinaryMERA.jl")

# Global variables in all-caps
const LAYERS = 8;
const CHI = 3;
const LIST_OF_CHIS = fill(CHI, LAYERS+2) 
M = generate_random_MERA(LIST_OF_CHIS);

ARANDOMOP = random_complex_tensor(CHI, 6) ;
THREESITEEYE = complex(ncon((eye(CHI),eye(CHI),eye(CHI)),([-1,-11], [-2,-12], [-3,-13])));

for i in 1:LAYERS
    # threesite operators
    randomop1 = random_complex_tensor(CHI, 3+3)
    randomop2 = random_complex_tensor(CHI, 3+3)
    thisLayer = M.levelTensors[i]

    # #LEFT TEST
    # down1 = binary_descend_threesite_left_c1(randomop1, thisLayer)
    # up2 = binary_ascend_threesite_left_c3(randomop2, thisLayer)
    # #RIGHT TEST
    # down1 = binary_descend_threesite_right_c1(randomop1, thisLayer)
    # up2 = binary_ascend_threesite_right_c3(randomop2, thisLayer)


    down1 = descend_threesite_symm(randomop1, thisLayer)
    up2 = ascend_threesite_symm(randomop2, thisLayer)

    e1 = ncon((down1, randomop2), ([1,2,3,11,12,13], [11,12,13,1,2,3]))[1]
    e2 = ncon((up2, randomop1), ([1,2,3,11,12,13], [11,12,13,1,2,3]))[1]
    @show abs(e1 - e2) / abs(e1)
    #@show abs(e1)
end

rlist = buildReverseRhosList(M) |> reverse;
# -2 because we disregarding the state above the levelTensors of the topLayer
for i in 0:(length(rlist)-2)
    expectationvalue1 = expectation(ascendTo(THREESITEEYE,M,i), rlist[i+1])
    expectationvalue2 = expectation(ascendTo(ARANDOMOP,M,i), rlist[i+1])
    println("E1 = ",expectationvalue1,"\t \& E2 = ",expectationvalue2)
end

# ------------------------------------------------------------
# PROFILING ASCENDING SUPEROPERATORS
# ------------------------------------------------------------

using Plots
pyplot()


MIN_A = 4
MAX_A = 14

CHISIZES_A = collect(MIN_A:MAX_A)
TIMES_A1 = zeros(MIN_A:MAX_A) |> float;

for chi in CHISIZES_A
    l1 = generate_random_layer(chi,chi);
    #chieye = complex(eye(chi));
    #threesiteeye = ncon((chieye, chieye, chieye), ([-1,-11], [-2,-12], [-3,-13]));
    #@time ( ascend_threesite_right(threesiteeye(chi),l1 ));
    TIMES_A1[chi-(MIN_A-1)] = @elapsed ( ascend_threesite_symm(threesiteeye(chi),l1));
end


function costscalingA(b::Int,a::Int)
    return ( log(TIMES_A1[a-(MIN_A-1)])-log(TIMES_A1[b-(MIN_A-1)]) ) / ( log(a) - log(b) );
end

SCalingpowerA = costscalingA(MAX_A-3,MAX_A)
MEmorycostA = sizeof(Complex{Float64})*(MAX_A^floor(SCalingpowerA)) / (1024^3)

println("Power of chi scaling: ", SCalingpowerA)
#println("Naive estimate of max memory cost: ", MEmorycostA," GB")

pA=plot(CHISIZES_A, TIMES_A1,
		line=(:dot,1), marker=(:hex,6), leg=false,
		xaxis=("Bond dimension (chi)",:log), yaxis=("Computational time (seconds)",:log),
		title=("Complexity of contraction algorithm for A"))

savefig(pA,"cost-scaling-A-single.png")

# ------------------------------------------------------------
# PROFILING DESCENDING SUPEROPERATORS
# ------------------------------------------------------------

MIN_D = 4
MAX_D = 14

CHISIZES_D = collect(MIN_D:MAX_D)
TIMES_D1 = zeros(MIN_D:MAX_D) |> float;

for chi in CHISIZES_D
     l1 = generate_random_layer(chi,chi);
     #chieye = complex(eye(chi));
     #threesiteeye = ncon((chieye, chieye, chieye), ([-1,-11], [-2,-12], [-3,-13]));
     TIMES_D1[chi-(MIN_D-1)] = @elapsed ( descend_threesite_symm(threesiteeye(chi),l1));
end

function costscalingD(b::Int,a::Int)
    return ( log(TIMES_D1[a-(MIN_D-1)])-log(TIMES_D1[b-(MIN_D-1)]) ) / ( log(a) - log(b) );
end

SCalingpowerD = costscalingD(MAX_D-3,MAX_D)
MEmorycostD = sizeof(Complex{Float64})*(MAX_D^floor(SCalingpowerD)) / (1024^3)

println("Power of chi scaling: ", SCalingpowerD)
#println("Naive estimate of max memory cost: ", MEmorycostD," GB")

pD=plot(CHISIZES_D, TIMES_D1,
		line=(:dot,1), marker=(:hex,6), leg=false,
		xaxis=("Bond dimension (chi)",:log), yaxis=("Computational time (seconds)",:log),
		title=("Complexity of contraction algorithm for D"))

savefig(pD,"cost-scaling-D-single.png")



