using Test
using OrdinaryDiffEq
using SteadyStateDiffEq
using Distributions
using FiniteStateProjection
using Catalyst
using SparseArrays
using LinearAlgebra
using Sundials
using Random

marg(vec; dims) = dropdims(sum(vec; dims); dims)
Random.seed!(100)

rs = @reaction_network begin
    σ_on * (1 - G), 0 --> G
    σ_off, G --> 0
    ρ_m, G --> G + M
    δ_m, M --> 0
    ρ_p, M --> M + P
    δ_p, P --> 0
end

sys = FSPSystem(rs, [:G, :M, :P])

prs = [exp(2 * randn()), exp(2 * randn())]
pmap = [
    :σ_on => rand(Gamma()),
    :σ_off => rand(Gamma()),
    :ρ_m => prs[1],
    :δ_m => prs[1] / exp(2 * randn()),
    :ρ_p => prs[2],
    :δ_p => prs[2] / exp(2 * randn()),
]

ps = last.(pmap)

Mmax = 20
Pmax = 80

u0 = zeros(2, Mmax + 1, Pmax + 1)
u0[1] = 1.0

tt = [1.0, 10.0, 50.0]

prob = ODEProblem(sys, u0, 50.0, pmap)
sol = solve(prob, Vern7(), abstol = 1.0e-6, saveat = tt)

fspmean(xx) = dot(xx, 0:(length(xx) - 1))

mG_asym = ps[1] / (ps[1] + ps[2])

for (i, t) in enumerate(tt)
    mG = mG_asym * (1 - exp(-(ps[1] + ps[2]) * t))
    mM = ps[3] * mG_asym *
        (
        1 / ps[4] * (1 - exp(-ps[4] * t)) -
            1 / (ps[4] - ps[1] - ps[2]) * (exp(-(ps[1] + ps[2]) * t) - exp(-ps[4] * t))
    )
    mP = ps[5] * ps[3] * mG_asym *
        (
        1 / (ps[4] * ps[6]) * (1 - exp(-ps[6] * t)) -
            1 / (ps[4] * (ps[6] - ps[4])) * (exp(-ps[4] * t) - exp(-ps[6] * t)) -
            1 / ((ps[6] - ps[1] - ps[2]) * (ps[4] - ps[1] - ps[2])) *
            (exp(-(ps[1] + ps[2]) * t) - exp(-ps[6] * t)) +
            1 / ((ps[6] - ps[4]) * (ps[4] - ps[1] - ps[2])) *
            (exp(-ps[4] * t) - exp(-ps[6] * t))
    )

    @test marg(sol.u[i], dims = (2, 3)) ≈ pdf.(Bernoulli(mG), 0:1) atol = 1.0e-4
    @test fspmean(marg(sol.u[i], dims = (1, 3))) ≈ mM atol = 1.0e-2
    @test fspmean(marg(sol.u[i], dims = (1, 2))) ≈ mP atol = 1.0e-2
end

A = SparseMatrixCSC(sys, size(u0), pmap, 0)
f = (du, u, p, t) -> mul!(vec(du), A, vec(u))

probA = ODEProblem(f, u0, 50.0)
solA = solve(probA, Vern7(), abstol = 1.0e-6, saveat = tt)

@test sol.u[1] ≈ solA.u[1] atol = 1.0e-4
@test sol.u[2] ≈ solA.u[2] atol = 1.0e-4
@test sol.u[3] ≈ solA.u[3] atol = 1.0e-4

## Steady-State Tests

#prob_ss = SteadyStateProblem(sys, u0, pmap)
#sol_ss = solve(prob_ss, DynamicSS(Vern7()))
#sol_ss.u ./= sum(sol_ss.u)
#
#@test marg(sol_ss.u, dims=2) ≈ pdf.(Poisson(ps[1] / ps[2]), 0:Nmax) atol=1e-4
#@test marg(sol_ss.u, dims=1) ≈ pdf.(Poisson(ps[3] / ps[4]), 0:Nmax) atol=1e-4
#
A_ss = SparseMatrixCSC(sys, size(u0), pmap, SteadyState())
#f_ss = (du,u,p,t) -> mul!(vec(du), A_ss, vec(u))
#
#probA_ss = SteadyStateProblem(f_ss, u0)
#solA_ss = solve(probA_ss, DynamicSS(Vern7()))
#solA_ss.u ./= sum(solA_ss.u)
#
#@test sol_ss.u ≈ solA_ss.u atol=1e-4

## Permutation tests

sys_perm = FSPSystem(rs, [:P, :M, :G])

pdims(x) = permutedims(x, (3, 2, 1))
u0_perm = pdims(u0)

A_perm = SparseMatrixCSC(sys_perm, (Pmax + 1, Mmax + 1, 2), pmap, 0)

idx_perm = vec(pdims(reshape(1:prod(size(u0)), size(u0))))
P = sparse(1:prod(size(u0)), idx_perm, 1)

@test A_perm ≈ P * A * P'

prob_perm = ODEProblem(sys_perm, u0_perm, 50.0, pmap)
sol_perm = solve(prob_perm, Vern7(), abstol = 1.0e-6, saveat = tt)

@test sol_perm.u[1] ≈ pdims(sol.u[1]) atol = 1.0e-4
@test sol_perm.u[2] ≈ pdims(sol.u[2]) atol = 1.0e-4
@test sol_perm.u[3] ≈ pdims(sol.u[3]) atol = 1.0e-4

## Steady-State Tests

A_fsp_ss_perm = SparseMatrixCSC(sys_perm, size(u0_perm), pmap, SteadyState())

@test A_fsp_ss_perm ≈ P * A_ss * P'

#prob_ss_perm = SteadyStateProblem(sys_perm, u0_perm, pmap)
#sol_ss_perm = solve(prob_ss_perm, SSRootfind())
#sol_ss_perm.u ./= sum(sol_ss_perm.u)
#
#@test sol_ss_perm.u ≈ pdims(sol_ss.u) atol=1e-4
