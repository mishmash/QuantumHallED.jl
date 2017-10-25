# include("AngularMomentum.jl")
# include("Pseudopotentials.jl")

using Combinatorics

abstract type HaldaneSphereStatesList end

struct HaldaneSphereStatesListLLL <: HaldaneSphereStatesList
    N::Int
    Nphi::Int
    Lz::AngMom

    possible_mz::Vector{Float64}

    orblist::Vector{Vector{Int}}
    nlist::Vector{Vector{Int}} # FIXME: use BitVector
    Ilist::Vector{Int}

    function HaldaneSphereStatesListLLL(N::Int, Nphi::Int, Lz::AngMom)
        Norb = Nphi + 1

        @assert N <= Norb
        @assert isangmom(Lz)

        Q = Nphi/2
        possible_mz = -Q:Q

        println("Building states list for N = $(N), Nphi = $(Nphi), Lz = $(Lz) ...")
        tic()

        orblist = Vector{Int}[] # occupied orbitals (ordered set containing N elements from 1:Norb)
        nlist = Vector{Int}[] # bit string of orbital occupations
        Ilist = Int[] # integer representative of state
        for orbs in combinations(1:Norb, N)
            if sum(possible_mz[orbs]) == Lz
                # orblist
                @assert issorted(orbs)
                push!(orblist, orbs)
                # nlist
                ns = nvec(orbs, Norb)
                push!(nlist, ns)
                # Ilist
                push!(Ilist, I(ns))
            end
        end

        sort_inds = sortperm(Ilist)
        orblist = orblist[sort_inds]
        nlist = nlist[sort_inds]
        Ilist = Ilist[sort_inds]

        println("Finished building states list of size $(length(Ilist))")
        toc()
        println()

        return new(N, Nphi, Lz, possible_mz, orblist, nlist, Ilist)
    end
end

get_N(states::HaldaneSphereStatesListLLL) = states.N
get_Nphi(states::HaldaneSphereStatesListLLL) = states.Nphi
get_Norb(states::HaldaneSphereStatesListLLL) = states.Nphi + 1
get_ell(states::HaldaneSphereStatesListLLL) = states.Nphi/2 # == angular momentum in the effective LLL problem == monopole strength Q in LLL

dim(states::HaldaneSphereStatesList) = length(states.Ilist)

mzvec(states::HaldaneSphereStatesListLLL, j::Int) = states.possible_mz[states.orblist[j]]

function nvec(orbs::Vector{Int}, Norb::Int)
    ns = zeros(Int, Norb)
    ns[orbs] = 1
    return ns
end

I(ns::Vector{Int}) = sum([ns[alpha]*2^(alpha-1) for alpha in 1:length(ns)])

ph_conjugate(ns::Vector{Int}) = 1 - ns

function ph_conjugate{T}(psi::Vector{T}, states::HaldaneSphereStatesListLLL)
    @assert length(psi) == dim(states)
    psi_ph = zeros(T, length(psi))
    for i in 1:length(psi)
        psi_ph[i] = psi[searchsorted(states.Ilist, I(ph_conjugate(states.nlist[i])))[1]]
    end
    return psi_ph
end


function build_Lsquared(states::HaldaneSphereStatesListLLL)
    println("Building L^2 matrix of size $(dim(states)) x $(dim(states)) ...")
    tic()

    M = get_Norb(states)
    Q = get_ell(states)

    rows = Int[]
    cols = Int[]
    LpLmvals = Float64[]
    LmLpvals = Float64[]
    @inbounds begin
    for j in 1:dim(states)
        if j == 1 || mod(j, ceil(0.05dim(states))) == 0
            println("Working on LpLm, column j = $(j): $(j)/$(dim(states)) = $(100j/dim(states))%")
        end
        ns = states.nlist[j]
        # diagonal terms of LpLm and LmLp
        push!(rows, j)
        push!(cols, j)
        push!(LpLmvals, sum([Aplus(Q, m(alpha, Q) - 1) * Aminus(Q, m(alpha, Q)) * ns[alpha] for alpha in 1:M])) # FIXME: don't use list comprehension
        push!(LmLpvals, sum([Aminus(Q, m(alpha, Q) + 1) * Aplus(Q, m(alpha, Q)) * ns[alpha] for alpha in 1:M])) # FIXME: don't use list comprehension
        # off-diagonal terms of LpLm and LmLp (they are identical)
        for alpha1 in 1:M-1
            for alpha2 in 2:M
                if (alpha1 != alpha2) && (alpha1 + 1 != alpha2 - 1)
                    if ns[alpha1] == 1 && ns[alpha2] == 1
                        sgn = alpha1 < alpha2 ? 1 : -1
                        ns_new = copy(ns)
                        ns_new[alpha1] = 0
                        ns_new[alpha2] = 0
                        if ns_new[alpha1 + 1] == 0 && ns_new[alpha2 - 1] == 0
                            sgn *= alpha1 + 1 < alpha2 - 1 ? 1 : -1
                            val = Aplus(Q, m(alpha1, Q)) * Aminus(Q, m(alpha2, Q))
                            if val != 0
                                p1 = sum(ns[1:alpha1-1])
                                p2 = sum(ns[1:alpha2-1])
                                p1p1 = sum(ns_new[1:(alpha1+1)-1])
                                p2m1 = sum(ns_new[1:(alpha2-1)-1])
                                sgn *= (-1)^(p1 + p2 + p1p1 + p2m1) # sign of matrix element from anticommutation relations
                                ns_new[alpha1 + 1] = 1
                                ns_new[alpha2 - 1] = 1
                                i = searchsorted(states.Ilist, I(ns_new))[1] # binary search to figure out which row to populate
                                push!(rows, i)
                                push!(cols, j)
                                push!(LpLmvals, -sgn * val)
                                push!(LmLpvals, -sgn * val)
                            end
                        end
                    end
                end
            end
        end
    end
    end
    LpLm = sparse(rows, cols, LpLmvals, dim(states), dim(states))
    LmLp = sparse(rows, cols, LmLpvals, dim(states), dim(states))

    cols = Int[]
    Lz2vals = Float64[]
    for j in 1:dim(states)
        if j == 1 || mod(j, ceil(0.10dim(states))) == 0
            println("Working on Lz^2, column j = $(j): $(j)/$(dim(states)) = $(100j/dim(states))%")
        end
        val = float(sum([m(alpha, Q) * states.nlist[j][alpha] for alpha in 1:M]))^2
        if val != 0
            push!(cols, j)
            push!(Lz2vals, val)
        end
    end
    Lz2 = sparse(cols, cols, Lz2vals, dim(states), dim(states))

    L2 = 0.5 * (LpLm + LmLp) + Lz2

    toc()
    println()

    return L2
end


abstract type HaldaneSphereSetup end

struct HaldaneSphereSetupLLL <: HaldaneSphereSetup
    states::HaldaneSphereStatesListLLL
    Lsquared::SparseMatrixCSC{Float64, Int64}

    function HaldaneSphereSetupLLL(N::Int, Nphi::Int, Lz::AngMom)
        # FIXME: add some asserts

        states = HaldaneSphereStatesListLLL(N, Nphi, Lz)
        Lsquared = build_Lsquared(states)

        return new(states, Lsquared)
    end
end

get_N(setup::HaldaneSphereSetupLLL) = get_N(setup.states)
get_Nphi(setup::HaldaneSphereSetupLLL) = get_Nphi(setup.states)
get_Norb(setup::HaldaneSphereSetupLLL) = get_Norb(setup.states)
get_ell(setup::HaldaneSphereSetupLLL) = get_ell(setup.states)

dim(setup::HaldaneSphereSetupLLL) = dim(setup.states)

Lqn(v) = 0.5 * (-1 + sqrt(1 + 4v)) # v = L * (L + 1)

function eig_Lsquared(Lsquared::SparseMatrixCSC{Float64, Int64})
    @assert size(Lsquared)[1] == size(Lsquared)[2]
    println("Diagonalizing L^2 matrix of size $(size(Lsquared)[1]) x $(size(Lsquared)[2]) with eig ...")
    tic()
    evals, evecs = eig(full(Lsquared))
    toc()
    println()
    return Lqn(evals), evecs
end

eig_Lsquared(setup::HaldaneSphereSetupLLL) = eig_Lsquared(setup.Lsquared)

# eigs on Lsquared is sketchy; use with caution..
function eigs_Lsquared(Lsquared::SparseMatrixCSC{Float64, Int64}, nev::Int)
    @assert size(Lsquared)[1] == size(Lsquared)[2]
    println("Diagonalizing L^2 matrix of size $(size(Lsquared)[1]) x $(size(Lsquared)[2]) with eigs, nev = $(nev) ...")
    tic()
    evals, evecs = eigs(Lsquared; which=:SR, nev=nev)
    toc()
    println()
    return Lqn(evals), evecs
end

eigs_Lsquared(setup::HaldaneSphereSetupLLL) = eigs_Lsquared(setup.Lsquared)


abstract type HaldaneSphereHami end

mutable struct HaldaneSphereHamiLLL <: HaldaneSphereHami
    setup::HaldaneSphereSetupLLL

    Vmat::Array{Float64, 4}
    H::SparseMatrixCSC{Float64, Int}

    energies::Vector{Float64}
    eigenstates::Matrix{Float64}

    spectrum::Matrix{Float64} # [L, E] pairs as rows in a 2-column matrix

    # use Vmat as input (most general case)
    function HaldaneSphereHamiLLL(setup::HaldaneSphereSetupLLL, Vmat::Array{Float64, 4})
        Norb = get_Norb(setup)
        @assert size(Vmat)[1] == size(Vmat)[2] == size(Vmat)[3] == size(Vmat)[4]
        @assert size(Vmat)[1] == Norb

        println("Building Hami matrix of size $(dim(setup)) x $(dim(setup)) ...")
        tic()
        rows = Int[]
        cols = Int[]
        Hvals = Float64[]
        @inbounds begin
        for j in 1:dim(setup)
            if j == 1 || mod(j, ceil(0.05dim(setup))) == 0
                println("Working on H, column j = $(j): $(j)/$(dim(setup)) = $(100j/dim(setup))%")
            end
            ns = setup.states.nlist[j]
            for alpha2 in 1:Norb
                for alpha1 in 1:alpha2-1
                    for alpha3 in 1:Norb
                        for alpha4 in 1:alpha3-1
                            if alpha1 + alpha2 == alpha3 + alpha4 # Lz conservation [mz = alpha - (Q + 1) = alpha + const.]
                                @assert alpha1 < alpha2 && alpha4 < alpha3
                                if ns[alpha3] == 1 && ns[alpha4] == 1
                                    ns_new = copy(ns)
                                    ns_new[alpha3] = 0
                                    ns_new[alpha4] = 0
                                    if ns_new[alpha1] == 0 && ns_new[alpha2] == 0
                                        V1234 = Vmat[alpha1, alpha2, alpha3, alpha4]
                                        V1243 = Vmat[alpha1, alpha2, alpha4, alpha3]
                                        V2134 = Vmat[alpha2, alpha1, alpha3, alpha4]
                                        V2143 = Vmat[alpha2, alpha1, alpha4, alpha3]
                                        val = V1234 - V1243 - V2134 + V2143
                                        if val != 0
                                            p1 = sum(ns_new[1:alpha1-1])
                                            p2 = sum(ns_new[1:alpha2-1])
                                            p3 = sum(ns[1:alpha3-1])
                                            p4 = sum(ns[1:alpha4-1])
                                            sgn = (-1)^(p1 + p2 + p3 + p4) # sign of matrix element from anticommutation relations
                                            ns_new[alpha1] = 1
                                            ns_new[alpha2] = 1
                                            i = searchsorted(setup.states.Ilist, I(ns_new))[1] # binary search to figure out which row to populate
                                            push!(rows, i)
                                            push!(cols, j)
                                            push!(Hvals, sgn * 0.5val)
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
        end
        H = sparse(rows, cols, Hvals, dim(setup), dim(setup))
        toc()
        println()

        return new(setup, Vmat, H)
    end
end

# compute Clebsch–Gordan coefficients from scratch
function HaldaneSphereHamiLLL(setup::HaldaneSphereSetupLLL, pp::SphericalPseudopotentials)
    return HaldaneSphereHamiLLL(setup, build_Vmat(pp, build_CG_table(get_ell(setup))))
end

# precomputed Clebsch–Gordan coefficients as input
function HaldaneSphereHamiLLL(setup::HaldaneSphereSetupLLL, pp::SphericalPseudopotentials, CG_table::Array{Vector{Float64}, 3})
    return HaldaneSphereHamiLLL(setup, build_Vmat(pp, CG_table))
end

# compute Clebsch–Gordan coefficients from scratch
function HaldaneSphereHamiLLL(N::Int, Nphi::Int, Lz::AngMom, pp::SphericalPseudopotentials)
    return HaldaneSphereHamiLLL(HaldaneSphereSetupLLL(N, Nphi, Lz), build_Vmat(pp, build_CG_table(get_ell(setup))))
end

# precomputed Clebsch–Gordan coefficients as input
function HaldaneSphereHamiLLL(N::Int, Nphi::Int, Lz::AngMom, pp::SphericalPseudopotentials, CG_table::Array{Vector{Float64}, 3})
    return HaldaneSphereHamiLLL(HaldaneSphereSetupLLL(N, Nphi, Lz), build_Vmat(pp, CG_table))
end

get_N(Hami::HaldaneSphereHamiLLL) = get_N(Hami.setup)
get_Nphi(Hami::HaldaneSphereHamiLLL) = get_Nphi(Hami.setup)
get_Norb(Hami::HaldaneSphereHamiLLL) = get_Norb(Hami.setup)
get_ell(Hami::HaldaneSphereHamiLLL) = get_ell(Hami.setup)

dim(Hami::HaldaneSphereHamiLLL) = dim(Hami.setup)

function eig!(Hami::HaldaneSphereHamiLLL)
    println("Diagonalizing Hami of size $(dim(Hami)) x $(dim(Hami)) with eig ...")
    tic()
    Hami.energies, Hami.eigenstates = eig(full(Hami.H))
    toc()
    println()
    return Hami.energies, Hami.eigenstates
end

function eigs!(Hami::HaldaneSphereHamiLLL, nev::Int)
    println("Diagonalizing Hami of size $(dim(Hami)) x $(dim(Hami)) with eigs, nev = $(nev) ...")
    tic()
    Hami.energies, Hami.eigenstates = eigs(Hami.H; which=:SR, nev=nev)
    toc()
    println()
    return Hami.energies, Hami.eigenstates
end

function organize_spectrum!(Hami::HaldaneSphereHamiLLL)
    println("Organizing spectrum according to quantum number L ...")
    tic()
    Hami.spectrum = zeros(Float64, (length(Hami.energies), 2))
    for j in 1:length(Hami.energies)
        Hami.spectrum[j, 1] = Lqn((Hami.eigenstates[:, j]' * Hami.setup.Lsquared * Hami.eigenstates[:, j])[1])
        Hami.spectrum[j, 2] = Hami.energies[j]
    end
    toc()
    println()
    return Hami.spectrum
end


function entanglement_spectrum{T}(psi::Vector{T}, states::HaldaneSphereStatesListLLL, subsystemA::Vector{Int}, NA::Int, LzAvec::Vector{Float64})
    @assert length(psi) == dim(states)

    IAlist = zeros(Int, length(psi))
    NAlist = zeros(Int, length(psi))
    LzAlist = zeros(Float64, length(psi))
    mzA = states.possible_mz[subsystemA]
    for i in 1:dim(states)
        nsA = states.nlist[i][subsystemA]
        IAlist[i] = I(nsA)
        NAlist[i] = sum(nsA)
        LzAlist[i] = vecdot(mzA, nsA)
    end

    println("Calculating entanglement spectrum for NA = $(NA), LzA = $(LzAvec) ...")
    tic()
    spec = zeros(Float64, (0, 2))
    for LzA in LzAvec

        # strip and sort
        strip_inds = intersect(find(NAlist .== NA), find(LzAlist .== LzA))
        IAlist_stripped = IAlist[strip_inds]
        sort_inds = sortperm(IAlist_stripped)
        IAlist_stripped = IAlist_stripped[sort_inds]
        # FIXME: check IB consistency
        psi_stripped = psi[strip_inds][sort_inds]

        # reshape psi into a matrix
        nc = length(find(IAlist_stripped .== IAlist_stripped[1]))
        nr = Int(length(psi_stripped) / nc)
        psi_matrix = reshape(psi_stripped, (nc, nr)).'

        # compute and diagonalize rhoA
        rhoA = psi_matrix * psi_matrix'
        evals = eigvals(rhoA)
        xi = -log.(evals[find(evals .> 0)])

        spec = vcat(spec, hcat(fill(LzA, length(xi)), sort(xi)))
    end
    toc()
    println()

    return spec
end

function entanglement_spectrum{T}(psi::Vector{T}, states::HaldaneSphereStatesListLLL, subsystemA::Vector{Int}, NA::Int, LzA::Float64)
    entanglement_spectrum(psi, states, subsystemA, NA, [LzA])
end


# matrix representation of c_alpha1p^dagger c_alpha2p^dagger c_alpha2 c_alpha1 in occupation number basis (very similiar to Hami building -- could probably reuse code..)
function four_point_operator(states::HaldaneSphereStatesListLLL, alpha1p::Int, alpha2p::Int, alpha1::Int, alpha2::Int)
    # make the following asserts to simplify the logic below (should be all we need with a little algebra)
    @assert alpha1p < alpha2p
    @assert alpha2 < alpha1
    @assert alpha1p + alpha2p == alpha1 + alpha2 # Lz conservation

    rows = Int[]
    cols = Int[]
    op_vals = Int[]
    @inbounds begin
    for j in 1:dim(states)
        # if j == 1 || mod(j, ceil(0.05dim(states))) == 0
        #     println("Working on c_$(alpha1p)^† c_$(alpha2p)^† c_$(alpha2) c_$(alpha1), column j = $(j): $(j)/$(dim(states)) = $(100j/dim(states))%")
        # end
        ns = states.nlist[j]
        if ns[alpha1] == 1 && ns[alpha2] == 1
            ns_new = copy(ns)
            ns_new[alpha1] = 0
            ns_new[alpha2] = 0
            if ns_new[alpha1p] == 0 && ns_new[alpha2p] == 0
                p1 = sum(ns_new[1:alpha1p-1])
                p2 = sum(ns_new[1:alpha2p-1])
                p3 = sum(ns[1:alpha1-1])
                p4 = sum(ns[1:alpha2-1])
                sgn = (-1)^(p1 + p2 + p3 + p4) # sign of matrix element from anticommutation relations
                ns_new[alpha1p] = 1
                ns_new[alpha2p] = 1
                i = searchsorted(states.Ilist, I(ns_new))[1] # binary search to figure out which row to populate
                push!(rows, i)
                push!(cols, j)
                push!(op_vals, sgn)
            end
        end
    end
    end
    op = sparse(rows, cols, op_vals, dim(states), dim(states))

    return op
end

function four_point_corrs{T}(psi::Vector{T}, states::HaldaneSphereStatesListLLL)
    @assert length(psi) == dim(states)

    Norb = get_Norb(states)

    println("Computing 4-point functions ...")
    tic()
    corr_mat = zeros(T, (Norb, Norb, Norb, Norb))
    for alpha2p in 1:Norb
        println("Working on alpha2p = $(alpha2p): $(alpha2p)/$(Norb)")
        for alpha1p in 1:alpha2p-1
            for alpha1 in 1:Norb
                for alpha2 in 1:alpha1-1
                    if alpha1p + alpha2p == alpha1 + alpha2 # Lz conservation [mz = alpha - (Q + 1) = alpha + const.]
                        corr_mat[alpha1p, alpha2p, alpha1, alpha2] = psi' * four_point_operator(states, alpha1p, alpha2p, alpha1, alpha2) * psi
                    end
                end
            end
        end
    end
    toc()
    println()

    return corr_mat
end

function pair_correlation_function(psi::Vector, states::HaldaneSphereStatesListLLL,
                                   θ1::Vector{Float64}, ϕ1::Vector{Float64}, θ2::Vector{Float64}, ϕ2::Vector{Float64};
                                   n::Int = 0)
    @assert length(psi) == dim(states)
    @assert length(θ1) == length(ϕ1) == length(θ2) == length(ϕ2)
    @assert n >= 0

    ell = get_ell(states)
    Q = ell - n # to use orbitals corresponding to Landau level index n
    Norb = get_Norb(states)

    corr_mat = four_point_corrs(psi, states)

    println("Computing pair correlation function for $(length(θ1)) points ...")
    tic()
    g = zeros(Complex128, length(θ1))
    for i in 1:length(g)
        if i == 1 || mod(i, ceil(0.05length(g))) == 0
            println("Working on g[i], i = $(i): $(i)/$(length(g)) = $(100i/length(g))%")
        end
        for alpha2p in 1:Norb
            for alpha1p in 1:alpha2p-1
                for alpha1 in 1:Norb
                    for alpha2 in 1:alpha1-1
                        g1 = (conj(monopole_harmonic(Q, ell, m(alpha1p, ell), θ1[i], ϕ1[i]))
                            * conj(monopole_harmonic(Q, ell, m(alpha2p, ell), θ2[i], ϕ2[i]))
                            * monopole_harmonic(Q, ell, m(alpha2, ell), θ2[i], ϕ2[i])
                            * monopole_harmonic(Q, ell, m(alpha1, ell), θ1[i], ϕ1[i]))
                        g2 = (conj(monopole_harmonic(Q, ell, m(alpha1p, ell), θ1[i], ϕ1[i]))
                            * conj(monopole_harmonic(Q, ell, m(alpha2p, ell), θ2[i], ϕ2[i]))
                            * monopole_harmonic(Q, ell, m(alpha1, ell), θ2[i], ϕ2[i])
                            * monopole_harmonic(Q, ell, m(alpha2, ell), θ1[i], ϕ1[i]))
                        g3 = (conj(monopole_harmonic(Q, ell, m(alpha2p, ell), θ1[i], ϕ1[i]))
                            * conj(monopole_harmonic(Q, ell, m(alpha1p, ell), θ2[i], ϕ2[i]))
                            * monopole_harmonic(Q, ell, m(alpha2, ell), θ2[i], ϕ2[i])
                            * monopole_harmonic(Q, ell, m(alpha1, ell), θ1[i], ϕ1[i]))
                        g4 = (conj(monopole_harmonic(Q, ell, m(alpha2p, ell), θ1[i], ϕ1[i]))
                            * conj(monopole_harmonic(Q, ell, m(alpha1p, ell), θ2[i], ϕ2[i]))
                            * monopole_harmonic(Q, ell, m(alpha1, ell), θ2[i], ϕ2[i])
                            * monopole_harmonic(Q, ell, m(alpha2, ell), θ1[i], ϕ1[i]))
                        g[i] += (g1 - g2 - g3 + g4) * corr_mat[alpha1p, alpha2p, alpha1, alpha2]
                    end
                end
            end
        end
    end
    toc()
    println()

    return g
end

# evaulate along the equator using great circle (gc) distance (in units of ell0); g normalized to unity as r --> infty (divide by (density)^2)
# returns r/ell0 and g as column vectors
function pair_correlation_function_equator_gc(psi::Vector, states::HaldaneSphereStatesListLLL; pts::Int = 200, frac::Float64 = 0.5)
    ϕ = linspace(0, frac * 2pi, pts)
    g = pair_correlation_function(psi, setup.states,
                                  fill(pi/2, length(ϕ)),
                                  collect(ϕ),
                                  fill(pi/2, length(ϕ)),
                                  fill(0., length(ϕ)),
                                  n=0)
    g /= (get_N(states)/(4pi))^2 # density = N/(4pi)
    println("Throwing out imaginary part of g: maximum(abs, imag(g)) = $(maximum(abs, imag(g)))") 
    return hcat(sqrt(get_ell(states)) * ϕ, real(g))
end
