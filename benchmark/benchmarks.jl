using FiniteStateProjection, BenchmarkTools
using Catalyst
using OrdinaryDiffEqVerner: Vern7
using SparseArrays, LinearAlgebra

const SUITE = BenchmarkGroup()

rs = @reaction_network begin
    r1, 0 --> A
    r2, A --> 0
    s1, 0 --> B
    s2, B --> 0
end

pmap = [:r1 => 2.0, :r2 => 1.3, :s1 => 1.7, :s2 => 0.8]
Nmax = 45
u0 = zeros(Nmax + 1, Nmax + 1)
u0[1] = 1.0

# =============================================================================
# FSPSystem + sparse operator construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["FSPSystem"] = @benchmarkable FSPSystem($rs, [:A, :B])
sys = FSPSystem(rs, [:A, :B])
SUITE["construct"]["sparse_A"] = @benchmarkable SparseMatrixCSC(
    $sys, ($Nmax + 1, $Nmax + 1), $pmap, 0
)

# =============================================================================
# FSP ODE problem + solve
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

SUITE["solve"]["ode_problem"] = @benchmarkable ODEProblem(
    $sys, $u0, 10.0, $pmap
)

prob = ODEProblem(sys, u0, 10.0, pmap)

SUITE["solve"]["vern7"] = @benchmarkable solve(
    $prob, Vern7(); abstol = 1.0e-6, saveat = [0.25, 1.0, 10.0]
)
