#module CFD
# actually called ConformalData, but short name for ease of using import/reload during development

using Reexport
using BinaryMERA
using IsingHam
using LinearMaps
using NCon

export fixptOp
export cftCylinder, spectrumCylinder
export P3, P6, cftPlane

#include("IsingHam.jl")
#include("BinaryMERA.jl")

# Note that the MERA represents the system on the cylinder, so ascend the isingHam through the Asop
# sufficient number of times to reach the fixed point
# Then Diagonalize H_{cft}
# How about diagonalizing the momentum superop simultaneously?
# Maybe we could use one of the left or the right scaling superops exclusively, and that would include the effect of translation in the diagonalization?


"""
Takes as input an operator for the bottom layer, ascends it to just below the scale invariant layer, and then ascends through a scale invariant layer a fixed number of times.
"""
function fixptOp(m::MERA, op, nIter::Int64; eValShift::Float64=0.0)
    # THERE's ALREADY ONE IN BinaryMERA.jl?
    # 1. Get the required operator to diagonalize H + delta P
    # Get the SIL
    sil = m.topLayer.levelTensors
    prepOp = op -> ascendTo(op,m,length(m.levelTensors))
    # 2. Compute cftH and cftP
    tmp = op |> prepOp
    #old = tmp
    for i in 1:nIter
        # Due to the 2-> coarsegraining, system size doubles, and ground state energy must halve.
        # So 2x to preserve energies?
        tmp = Asop(sil)(tmp)
        # Do this for "sufficient" number of steps. What is a diagnostic for improvement?
    end

    return tmp
end

# ------------------------------------------------------------
# Conformal data from physics on the cylinder
# ------------------------------------------------------------

"""
Provides the Nsq*Nsq matrix to swap two neighboring sites with N dimensional Hilbert spaces.
"""
function swapTwoSites(N::Int64)
    # Initialize a matrix of size Nsq*Nsq
    swp = zeros(N*N,N*N)

    for i in 1:N
        for j in 1:N
            swp[N*(i-1)+j,N*(j-1)+i] = 1.
            swp[N*(j-1)+i,N*(i-1)+j] = 1.
            # The i==j case is automatically taken care of!
        end
    end

    return swp
end

"""
Returns a three site operator that moves each effective site to the next one.
"""
function P3(N::Int64)
    return (kron(swapTwoSites(N),eye(N))*kron(eye(N),swapTwoSites(N)))
end

"""
Returns a three site operator that moves each effective site to the right by half a site.
Effectively, one site translation on six sites, coarsegrained by one layer of the MERA.
"""
function P6(sil::Layer)
    (u, udag) = (sil.u.elem, sil.udag.elem)
    (w, wdag) = (sil.w.elem, sil.wdag.elem)

    # chi^{11} contraction order computed using 'netcon.m'
    P_op = ncon((w,w,w,     u,u,u,      udag,udag,udag,     wdag,wdag,wdag),
                ([-1,1,2], [-2,3,4], [-3,5,6],
                [2,3,12,13], [4,5,14,15], [6,1,16,11],
                [15,16,25,26], [11,12,21,22], [13,14,23,24],
                [26,21,-11], [22,23,-12], [24,25,-13]),
                order=[3,2,22,21,6,25,14,15,24,5,16,11,23,26,1,4,12,13] )
    return P_op
end

# Algorithmically generate P12 and in general P(3*2^k)
# with the interface P3(k::Int64) since it should produce a three-site operator
# implementing a one-site translation k layers deep

"""
Returns the tuple (H,P,S) of three site operators in the deep infrared, which could be used to find the spectrum of a CFT.
"""
function cftCylinder(m::MERA, chi::Int64, nIter::Int64)
    isingH, Dmax = build_H_Ising()
    H = fixptOp(m, isingH, nIter; eValShift=Dmax) |> imposePDBC
    #P = fixptOp(m, build_P_Ising())
    ## What about the representation of the spin flip operator?
    ## Does that stay unaffected by coarsegrainings? (manifests the Z2 symmetry of the model)
    #S = fixptOp(m, build_S_Ising())

    # P and S needn't be stepped up. Just construct them at the top layer!
    # P = P3(chi)
    sil = m.topLayer.levelTensors
    P = P6(sil)

    # This P is not fine enough to split all the eigenvalues nicely!
    # If the bond-dimension in the IR had a small factor (for example 8%2=0)
    # then we could have defined a much finer notion of translation.
    # Would be in/compatible with Hilbert space structure of Hamiltonian???

    #dim = size(H) |> prod |> sqrt |> Int
    dim = chi^3
    S = zeros(dim,dim)
    return map(x -> reshape(x, (dim, dim) ),(H,P,S))
end

"""
Returns the tuple (Scal,Spin) of three site SuperOperators in the deep infrared (corresponding to the scale invariant layer), which could be used to find the spectrum of a CFT.
"""
function spectrumCylinder(H,P,S)
    #return map(x -> reshape(x, (7,7,7,7,7,7) ),(H,P,S))
    D,U = eigs(H*(eye(H) + (1e-10)*P + (1e-12)*S), nev=12);

    return D
    #plot(imag(D), (real(D)+4*N)/N, marker = "o", linestyle =" " )
end

# ------------------------------------------------------------
# Conformal data from physics on the plane
# ------------------------------------------------------------

function cftPlane(sil::Layer)
    #N = 8 # Effectively modeled as a 3-site system in the IR

    function ScalSop(l::Layer)
        return (op::Vector{Complex128} -> ( reshape(op,(6,6,6,6,6,6)) |>
                        (x -> 0.5*ascend_threesite_left(x, l) + 0.5*ascend_threesite_right(x, l) ) |> vec ) )
    end

    function SpinSop(l::Layer)
        return (op::Vector{Complex128} -> ( reshape(op,(6,6,6,6,6,6)) |>
                        (x -> 0.5*ascend_threesite_left(x, l) - 0.5*ascend_threesite_right(x, l) ) |> vec ) )
    end

    Scal = LinearMap(ScalSop(sil), Complex128, 6^6, ishermitian=true , issym=false);
    Spin = LinearMap(SpinSop(sil), Complex128, 6^6, ishermitian=false, issym=false);
    # The eigenvalues of these must be exponentials of the eigenvalues on the cylinder

    # What about the representation of the spin flip operator?
    # Does that stay unaffected by coarsegrainings? (manifests the Z2 symmetry of the model)
    # Do we need to exponentiate it?
    #Sflip = fixptOp(sil, build_S_Ising())

    return (Scal,Spin)
end

# """
# Find the spectrum of a CFT, given three site operators in the deep infrared for (H,P,S).
# """

# Why should these two ways of computing conformal data give the same answer?
# The ASOP is _tuned_ for a particular Hamiltonian, so it encodes the same information as the critical Hamiltonian i.e. ascending a Hamiltonian infinitely many times and diagonalizing H_{cft} is morally similar to diagonalizing ASOP.

# After all, the fixedpoint Hamiltonian is simply a (suitable) linear combination of a it's eigenstates/operators, each of which must also be an eigenoperator of the Asop. (What about the weights... is the eigenvalue of the Asop an exponential of the energy?)

#end
