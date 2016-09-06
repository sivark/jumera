# Ising model with transverse magnetic field h (critical h=1 by default)
# Returns three-site Ising Hamiltonian (8x8 matrix), and the highest energy eigenvalue
function build_H_Ising(h=1.0)
    h::Array{Complex{Float64},6}
    D_max::Float64
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    I = eye(2)
    XX = kron(X,X)
    ZI = kron(Z,I)
    IZ = kron(I,Z)
    H2 = -(XX + h/2*(ZI+IZ))
    H = H2 / 3  # See below for an explanation of the 1/3.
    for n = 3:9
        eyen2 = eye(2^(n-2))
        # Terms at the borders of the blocks of three that get grouped together
        # need to be normalized differently from the ones that are within blocks.
        factor = (n == 4 || n == 7) ? 1/2 : 1/3
        H = kron(H, I) + kron(eyen2, H2)*factor
    end
    D, V = eig(Hermitian(H))
    D_max = D[end]
    # subtract largest eigenvalue, so that the spectrum is negative
    H = H - eye(2^9)*D_max
    h = reshape(H, (8,8,8,8,8,8)) |> complex
    return h, D_max
end

function approximate_energy_persite_PBC(nsites)
    # including only the leading finite-size correction
    return ( -4/pi - (pi/6)/(nsites*nsites) )
end

function exact_energy_persite(n_lyr)
    # Obtained by exact diagonalization of a free fermions system with the opposite BCs
    EnAPBC=[-1.270005811417927, -1.2724314193572888, -1.2730375326245706, -1.273189042909428, -1.2732269193538452, -1.2732363883945284];

    EnPBC_exactdiag_1_8=[-1.2797267740319183, -1.2748570272966502, -1.273643645891852, -1.273340553194287, -1.2732647957982595, -1.273245857435202, -1.273241122906045, -1.273239939277603];

    EnPBC_approx_9_15 = map(approximate_energy_persite_PBC,81*4.^collect(9:15));

    EnPBC=[EnPBC_exactdiag_1_8...,EnPBC_approx_9_15...]
    # Exact results only for up to 8 layers. Beyond that, 1/Nsq approximation is good enough since our MERA is not yet that accurate

    return EnPBC[n_lyr]
end
