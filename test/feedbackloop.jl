# Thanks to Xiaoming Fu for sharing his example code

using Test
using OrdinaryDiffEq
using SteadyStateDiffEq
using FiniteStateProjection
using Catalyst
using SparseArrays
using LinearAlgebra

rs = @reaction_network begin
    σ_off, G + P → 0
    σ_on * (1 - G), 0 ⇒ G + P
    ρ_on, G → G + P
    ρ_off * (1 - G), 0 ⇒ P
    d, P → 0
end

pmap = [
    :σ_off => 1,
    :σ_on => 0.1,
    :ρ_on => 20,
    :ρ_off => 1,
    :d => 1,
]

ps = last.(pmap)

Nmax = 200

sys = FSPSystem(rs, [:G, :P])

function create_A(ps, Nmax)
    σ_off, σ_on, ρ_on, ρ_off, d = ps

    A00 = sparse(
        [
            j == i ? -(ρ_off + σ_on + (i - 1) * d) :
                j == i - 1 ? ρ_off : j == i + 1 ? i * d : 0 for i in 1:Nmax, j in 1:Nmax
        ]
    )
    A01 = sparse([j == i + 1 ? i * σ_off : 0 for i in 1:Nmax, j in 1:Nmax])

    A10 = sparse([j == i - 1 ? σ_on : 0 for i in 1:Nmax, j in 1:Nmax])
    A11 = sparse(
        [
            j == i ? -(ρ_on + (i - 1) * (d + σ_off)) :
                j == i - 1 ? ρ_on : j == i + 1 ? i * d : 0 for i in 1:Nmax, j in 1:Nmax
        ]
    )

    A = SparseMatrixCSC{Float64, Int64}([A00 A01; A10 A11])

    # Convert to column-major ordering used by Julia
    J = vec(reshape(1:(2 * Nmax), 2, Nmax)')
    P = sparse(1:(2 * Nmax), J, 1)
    return P' * A * P
end

A = create_A(ps, Nmax)
A_fsp = SparseMatrixCSC(sys, (2, Nmax), pmap, 0.0)

@test A ≈ A_fsp

tt = [1.0, 10.0, 20.0]

u0 = zeros(2, Nmax)
u0[1] = 1.0

prob = ODEProblem(sys, u0, maximum(tt), pmap)
sol = solve(prob, Vern7(), abstol = 1.0e-6, saveat = tt)

f = (du, u, p, t) -> mul!(vec(du), A, vec(u))

probA = ODEProblem(f, u0, 20.0)
solA = solve(probA, Vern7(), abstol = 1.0e-6, saveat = tt)

@test sol.u[1] ≈ solA.u[1] atol = 1.0e-4
@test sol.u[2] ≈ solA.u[2] atol = 1.0e-4
@test sol.u[3] ≈ solA.u[3] atol = 1.0e-4

## Steady-State Tests

function create_A_ss(ps, Nmax)
    A = create_A(ps, Nmax)

    for i in 1:size(A, 1)
        A[i, i] -= sum(A[:, i])
    end

    return A
end

A_ss = create_A_ss(ps, Nmax)
A_fsp_ss = SparseMatrixCSC(sys, (2, Nmax), pmap, SteadyState())

@test A_ss ≈ A_fsp_ss

prob_ss = SteadyStateProblem(sys, u0, pmap)
sol_ss = solve(prob_ss, SSRootfind())
sol_ss.u ./= sum(sol_ss.u)

f_ss = (du, u, p, t) -> mul!(vec(du), A, vec(u))

probA_ss = SteadyStateProblem(f, u0)
solA_ss = solve(probA_ss, SSRootfind())
solA_ss.u ./= sum(solA_ss.u)

@test sol_ss.u ≈ solA_ss.u atol = 1.0e-4

### Permutations

sys_perm = FSPSystem(rs, [:P, :G])

u0_perm = u0'

A_perm = SparseMatrixCSC(sys_perm, (Nmax, 2), pmap, 0.0)

idx_perm = vec(reshape(1:(2 * Nmax), (Nmax, 2))')
P = sparse(1:(2 * Nmax), idx_perm, 1)'

@test A_perm ≈ P * A * P'

prob_perm = ODEProblem(sys_perm, u0_perm, maximum(tt), pmap)
sol_perm = solve(prob_perm, Vern7(), abstol = 1.0e-6, saveat = tt)

@test sol_perm.u[1] ≈ sol.u[1]' atol = 1.0e-4
@test sol_perm.u[2] ≈ sol.u[2]' atol = 1.0e-4
@test sol_perm.u[3] ≈ sol.u[3]' atol = 1.0e-4

A_fsp_ss_perm = SparseMatrixCSC(sys_perm, (Nmax, 2), pmap, SteadyState())

@test A_fsp_ss_perm ≈ P * A_ss * P'

prob_ss_perm = SteadyStateProblem(sys_perm, u0_perm, pmap)
sol_ss_perm = solve(prob_ss_perm, SSRootfind())
sol_ss_perm.u ./= sum(sol_ss_perm.u)

@test sol_ss_perm.u ≈ sol_ss.u' atol = 1.0e-4
