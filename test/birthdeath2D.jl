using Test
using AxisArrays: AxisArray, Axis
using OrdinaryDiffEq
using SteadyStateDiffEq
using Distributions
using FiniteStateProjection
using Catalyst
using SparseArrays
using LinearAlgebra
using Random
using Sundials

marg(vec; dims) = dropdims(sum(vec; dims); dims)

Random.seed!(1234)

rs = @reaction_network begin
    r1, 0 --> A
    r2, A --> 0
    s1, 0 --> B
    s2, B --> 0
end

sys = FSPSystem(rs, [:A, :B])

pmap = [
    :r1 => 2.0,
    :r2 => 1.3,
    :s1 => 1.7,
    :s2 => 0.8,
]

ps = last.(pmap)

Nmax = 45

u0 = zeros(Nmax + 1, Nmax + 1)
u0[1] = 1.0

tt = [0.25, 1.0, 10.0]

prob = ODEProblem(sys, u0, 10.0, pmap)
sol = solve(prob, Vern7(), abstol = 1.0e-6, saveat = tt)

@test marg(sol.u[1], dims = 2) ≈
    pdf.(Poisson(ps[1] / ps[2] * (1 - exp(-ps[2] * tt[1]))), 0:Nmax) atol = 1.0e-4
@test marg(sol.u[1], dims = 1) ≈
    pdf.(Poisson(ps[3] / ps[4] * (1 - exp(-ps[4] * tt[1]))), 0:Nmax) atol = 1.0e-4

@test marg(sol.u[2], dims = 2) ≈
    pdf.(Poisson(ps[1] / ps[2] * (1 - exp(-ps[2] * tt[2]))), 0:Nmax) atol = 1.0e-4
@test marg(sol.u[2], dims = 1) ≈
    pdf.(Poisson(ps[3] / ps[4] * (1 - exp(-ps[4] * tt[2]))), 0:Nmax) atol = 1.0e-4

@test marg(sol.u[3], dims = 2) ≈
    pdf.(Poisson(ps[1] / ps[2] * (1 - exp(-ps[2] * tt[3]))), 0:Nmax) atol = 1.0e-4
@test marg(sol.u[3], dims = 1) ≈
    pdf.(Poisson(ps[3] / ps[4] * (1 - exp(-ps[4] * tt[3]))), 0:Nmax) atol = 1.0e-4

A = SparseMatrixCSC(sys, (Nmax + 1, Nmax + 1), pmap, 0)
f = (du, u, p, t) -> mul!(vec(du), A, vec(u))

probA = ODEProblem(f, u0, 10.0)
solA = solve(probA, Vern7(), abstol = 1.0e-6, saveat = tt)

@test sol.u[1] ≈ solA.u[1] atol = 1.0e-4
@test sol.u[2] ≈ solA.u[2] atol = 1.0e-4
@test sol.u[3] ≈ solA.u[3] atol = 1.0e-4

## Steady-State Tests

prob_ss = SteadyStateProblem(sys, u0, pmap)
sol_ss = solve(prob_ss, SSRootfind())
sol_ss.u ./= sum(sol_ss.u)

@test marg(sol_ss.u, dims = 2) ≈ pdf.(Poisson(ps[1] / ps[2]), 0:Nmax) atol = 1.0e-4
@test marg(sol_ss.u, dims = 1) ≈ pdf.(Poisson(ps[3] / ps[4]), 0:Nmax) atol = 1.0e-4

A_ss = SparseMatrixCSC(sys, (Nmax + 1, Nmax + 1), pmap, SteadyState())
f_ss = (du, u, p, t) -> mul!(vec(du), A_ss, vec(u))

probA_ss = SteadyStateProblem(f_ss, u0)
solA_ss = solve(probA_ss, SSRootfind())
solA_ss.u ./= sum(solA_ss.u)

@test sol_ss.u ≈ solA_ss.u atol = 1.0e-4

## Permutation tests

sys_perm = FSPSystem(rs, [:B, :A])

u0_perm = u0'

A_perm = SparseMatrixCSC(sys_perm, (Nmax + 1, Nmax + 1), pmap, 0)

idx_perm = vec(reshape(1:((Nmax + 1)^2), (Nmax + 1, Nmax + 1))')
P = sparse(1:((Nmax + 1)^2), idx_perm, 1)'

@test A_perm ≈ P * A * P'

prob_perm = ODEProblem(sys_perm, u0_perm, 10.0, pmap)
sol_perm = solve(prob_perm, Vern7(), abstol = 1.0e-6, saveat = tt)

@test sol_perm.u[1] ≈ sol.u[1]' atol = 1.0e-4
@test sol_perm.u[2] ≈ sol.u[2]' atol = 1.0e-4
@test sol_perm.u[3] ≈ sol.u[3]' atol = 1.0e-4

## Steady-State Tests

A_fsp_ss_perm = SparseMatrixCSC(sys_perm, (Nmax + 1, Nmax + 1), pmap, SteadyState())

@test A_fsp_ss_perm ≈ P * A_ss * P'

prob_ss_perm = SteadyStateProblem(sys_perm, u0_perm, pmap)
sol_ss_perm = solve(prob_ss_perm, SSRootfind())
sol_ss_perm.u ./= sum(sol_ss_perm.u)

@test sol_ss_perm.u ≈ sol_ss.u' atol = 1.0e-4

## AxisArray template tests

template = AxisArray(zeros(Nmax + 1, Nmax + 1), Axis{:A}(0:Nmax), Axis{:B}(0:Nmax))
sys_tpl = FSPSystem(rs, template)

@test sys_tpl.ih == OrderedIndexHandler(rs, [:A, :B])
# The matrix is indexed by array positions, so it takes the size of the template.
@test SparseMatrixCSC(sys_tpl, size(template), pmap, 0) ≈ A
@test SparseMatrixCSC(sys_tpl, size(template), pmap, SteadyState()) ≈ A_ss
