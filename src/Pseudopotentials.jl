# include("AngularMomentum.jl")

abstract type Pseudopotentials end

mutable struct SphericalPseudopotentials <: Pseudopotentials
    ell::AngMom
    Q::AngMom

    L::AngMomVector # total angular momentum on sphere
    m::AngMomVector # == 2ell - L == relative angular momenta on sphere
    V::Vector{Float64} # the actual pseudopotentials
end

# use the expression in the thesis by Wooten: http://trace.tennessee.edu/cgi/viewcontent.cgi?article=2884&context=utk_graddiss
# have checked numerically that this is equivalent to the expression in Toke and Jain, PRL 96, 246805 (2006) [Eq. (6)]
# NB: this assumes chord distance separation on the sphere when defining the 2-body potential V(r1 - r2)
function SphericalPseudopotentials(ell::AngMom, Q::AngMom, Vk::Vector{Float64})
    @assert isangmom(ell) && isangmom(Q)
    @assert ell >= Q

    L = collect(0:2ell)
    V = zeros(Float64, length(L))
    for i in 1:length(L)
        Vtmp = 0.
        for k in 0:Int(2ell)
            Vtmp += Vk[k+1] * Wigner6j(L[i], ell, ell, k, ell, ell) * Wigner3j(ell, k, ell, -Q, 0, Q)^2
        end
        V[i] = Vtmp * (-1)^(2Q + L[i]) * (2ell + 1)^2
    end

    m = 2ell - L
    p = sortperm(m) # return L, m, and V according to increasing relative angular momentum m = 2ell - L

    return SphericalPseudopotentials(ell, Q, L[p], m[p], V[p])
end

# for pure Coulomb interaction, the parameters Vk = 1/(radius of sphere) = const. (in Coulomb units e^2/(ϵ * ell0))
function spherical_Coulomb_pseudopotentials(ell::AngMom, Q::AngMom, R::Float64)::SphericalPseudopotentials
    return SphericalPseudopotentials(ell, Q, fill(1/R, Int(2ell)+1))
end

# pseudopotentials for LL index n = 0, 1, ...
# work in units assuming that we are in an effective LLL problem with monopole strength ell (= Q)
# i.e., so that the radius of the sphere is sqrt(ell) (in units of the magnetic length ell0)
# using this function will put energies in Coulomb units e^2/(ϵ * ell0) for the effective LLL problem
function spherical_Coulomb_pseudopotentials(ell::AngMom, n::Int)::SphericalPseudopotentials
    return spherical_Coulomb_pseudopotentials(ell, ell - n, sqrt(ell))
end

function set_pseudopotentials!(pp::SphericalPseudopotentials, Vnew::Dict)
    for m in keys(Vnew)
        @assert isangmom(m)
        ind = find(pp.m .== m)
        if length(ind) == 0
            warn("m = $(m) not found in pp.m")
        end
        pp.V[ind] = convert(Float64, Vnew[m])
    end
    return pp
end

# dinosaur, but keep it in here for reference
function V_Coulomb(L::Int, Nphi::Int, R::Float64)
    @assert 0 <= L <= Nphi
    # the following closed-form expression blows up for large Q, so we use the 2nd expression (see notes)
    # return (2/R) * (binomial(Int(4Q - 2L), Int(2Q - L))
    #              *  binomial(Int(4Q + 2L + 2), Int(2Q + L + 1))
    #              /  binomial(Int(4Q + 2), Int(2Q + 1))^2)
    if L == 0
        return (1/R) * float((Nphi + 1)/(2Nphi + 1))
    else # L >= 1
        return (1/R) * (prod([(2*(Nphi + L + 1 - i) - 1)/(Nphi + L + 1 - i) for i in 0:L-1])
                     *  prod([(Nphi + 1 - i)/(2*(Nphi + 1 - i) - 1) for i in 0:L]))
    end
end

function build_Vmat(pp::SphericalPseudopotentials, CG_table::Array{Vector{Float64}, 3})
    Norb = length(pp.V)
    @assert size(CG_table)[1] == size(CG_table)[2] == size(CG_table)[3]
    @assert size(CG_table)[1] == Norb

    println("Building 2-body interaction matrix ...")
    tic()
    Vmat = zeros(Float64, (Norb, Norb, Norb, Norb))
    for alpha1 in 1:Norb
        println("Working on alpha1 = $(alpha1): $(alpha1)/$(Norb)")
        for alpha2 in 1:Norb
            for alpha3 in 1:Norb
                for alpha4 in 1:Norb
                    V = 0.
                    for L in 0:Int(2pp.ell)
                        for Mz in -L:L
                            @assert pp.L[end - L] == L
                            V += CG_table[alpha1, alpha2, L + 1][Mz + (L + 1)] * conj(CG_table[alpha3, alpha4, L + 1][Mz + (L + 1)]) * pp.V[end - L] 
                        end
                    end
                    Vmat[alpha1, alpha2, alpha3, alpha4] = V
                end
            end
        end
    end
    toc()
    println()

    return Vmat
end
