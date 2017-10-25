using PyCall
using JLD

# @pyimport sympy.physics.quantum.cg as cg
@pyimport sympy.physics.wigner as wigner

AngMom = Union{Int, Float64}
AngMomVector = Union{Vector{Int}, Vector{Float64}} # unfortunately this seems to be necessary
isangmom(q::AngMom) = isinteger(2q)
isangmom(qs::AngMomVector) = all(isinteger, 2qs)

m(alpha::Int, ell::AngMom)::Float64 = (alpha - 1) - ell

Aplus(ell::AngMom, m::AngMom)::Float64 = sqrt( ell * (ell + 1) - m * (m + 1) )
Aminus(ell::AngMom, m::AngMom)::Float64 = sqrt( ell * (ell + 1) - m * (m - 1) )

# CG(j1, m1, j2, m2, j3, m3) = (cg.CG(j1, m1, j2, m2, j3, m3)[:doit]() |> float)
CG(j1, m1, j2, m2, j3, m3) = (wigner.clebsch_gordan(j1, j2, j3, m1, m2, m3)[:doit]() |> float)
Wigner3j(j1, j2, j3, m1, m2, m3) = (wigner.wigner_3j(j1, j2, j3, m1, m2, m3)[:doit]() |> float)
Wigner6j(j1, j2, j3, j4, j5, j6) = (wigner.wigner_6j(j1, j2, j3, j4, j5, j6)[:doit]() |> float)

function build_CG_table(ell1::AngMom, ell2::AngMom)
    @assert isangmom(ell1)
    @assert isangmom(ell2)

    dim1 = Int(2ell1) + 1
    dim2 = Int(2ell2) + 1

    Ls = abs(ell1-ell2):ell1+ell2
    dim3 = length(Ls)

    println("Building CG table for ell1 = $(ell1), ell2 = $(ell2):")
    tic()
    CG_table = Array{Vector{Float64}}(dim1, dim2, dim3)
    for alpha1 in 1:dim1
        println("Working on alpha1 = $(alpha1): $(alpha1)/$(dim1)")
        for alpha2 in 1:dim2
            # println("Working on alpha2 = $(alpha2): $(alpha2)/$(dim2)")
            for alpha3 in 1:dim3
                tmp = Float64[]
                for M in -Ls[alpha3]:Ls[alpha3]
                    push!(tmp,  CG(float(ell1), m(alpha1, ell1), float(ell2), m(alpha2, ell2), Ls[alpha3], M))
                end
                CG_table[alpha1, alpha2, alpha3] = tmp
            end
        end
    end
    toc()
    println()

    return CG_table
end

build_CG_table(ell::AngMom) = build_CG_table(ell, ell)

function write_CG_table(ell1::AngMom, ell2::AngMom, dir::String)
    @assert isangmom(ell1)
    @assert isangmom(ell2)

    filename = "CG_$(float(ell1))_$(float(ell2)).jld"
    fullpath = joinpath(dir, filename)

    println("Writing CG table for ell1 = $(ell1), ell2 = $(ell2) to $(fullpath) ...")
    table = build_CG_table(ell1, ell2)
    save(fullpath, "table", table)
    println("Write successful")
end
write_CG_table(ell::AngMom, dir::String) = write_CG_table(ell, ell, dir)

function load_CG_table(ell1::AngMom, ell2::AngMom, dir::String)
    @assert isangmom(ell1)
    @assert isangmom(ell2)

    filename = "CG_$(float(ell1))_$(float(ell2)).jld"
    fullpath = joinpath(dir, filename)

    println("Loading CG table for ell1 = $(ell1), ell2 = $(ell2) from $(fullpath) ...")
    table = load(fullpath, "table")
    println("Load successful")
    return table
end
load_CG_table(ell::AngMom, dir::String) = load_CG_table(ell, ell, dir)

function compile_CG_tables(ell1vec::AngMomVector, ell2vec::AngMomVector, dir::String)
    for ell1 in ell1vec
        for ell2 in ell2vec
            write_CG_table(ell1, ell2, dir)
        end
    end
end

function compile_CG_tables(ellvec::AngMomVector, dir::String)
    for ell in ellvec
        write_CG_table(ell, dir)
    end
end

u(θ::Float64, ϕ::Float64)::Complex128 = cos(θ/2) * exp(im * ϕ/2)
v(θ::Float64, ϕ::Float64)::Complex128 = sin(θ/2) * exp(-im * ϕ/2)

function monopole_harmonic(Q::AngMom, ell::AngMom, m::AngMom, θ::Float64, ϕ::Float64)::Complex128
    @assert isangmom(Q)
    @assert isangmom(ell)
    @assert isangmom(m)

    @assert Q >= 0 # for simplicity
    @assert ell >= Q
    @assert abs(m) <= ell
    # could do more thorough consistency checks regarding integers versus half integers etc., but these will get caught below

    if m < Q
        N = prod([(ell - m - i)/(ell + Q - i) for i in 0:Q-m-1])
    elseif m > Q
        N = prod([(ell + m - i)/(ell - Q - i) for i in 0:m-Q-1])
    else
        N = 1.
    end
    N = sqrt((2ell + 1)/(4pi) * N)

    Y = sum([(-1.)^s * binomial(Int(ell-Q), s) * binomial(Int(ell+Q), Int(ell-m-s)) * sin(θ/2)^(2ell-2Q-2s) * cos(θ/2)^(2s) for s in 0:Int(ell-Q)])
    Y *= N * (-1.)^(ell - m) * v(θ, ϕ)^(Q - m) * u(θ, ϕ)^(Q + m)

    return Y
end
