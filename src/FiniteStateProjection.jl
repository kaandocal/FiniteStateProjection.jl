module FiniteStateProjection

using Catalyst: Reaction, ReactionSystem, jumpratelaw, netstoichmat, numspecies, reactions,
    species
import Catalyst
using ModelingToolkitBase: equations, get_systems, parameters
using PrecompileTools: @compile_workload, @setup_workload
import RuntimeGeneratedFunctions
using RuntimeGeneratedFunctions: @RuntimeGeneratedFunction
import SciMLBase
using SciMLBase: ODEFunction, ODEProblem, SteadyStateProblem
import SparseArrays
using SparseArrays: sparse
using SymbolicIndexingInterface: getname
using Symbolics: @variables, build_function
using SymbolicUtils: substitute

import Base: LinearIndices, vec

# ---------------------------------------------------------------------------
# Reexported interface (see the second `export` block below).
#
# The documented entry point of this package is a Catalyst reaction network:
#
#     rn = @reaction_network begin
#         σ, 0 --> A
#         d, A --> 0
#     end
#     sys = FSPSystem(rn)
#     prob = ODEProblem(sys, u0, (0, 10.0), ps)
#
# `@reaction_network` and the problem constructors reached the user through
# `@reexport using Catalyst`, so `using FiniteStateProjection` was enough to write
# that. The blanket reexport (441 names, most of them Catalyst's symbolic and
# modelling stack) is gone; this is the subset that normal documented use needs,
# each imported from the module that owns it and documented there.
# ---------------------------------------------------------------------------
using Catalyst: @reaction_network
using CommonSolve: solve

RuntimeGeneratedFunctions.init(@__MODULE__)

export AbstractIndexHandler, FSPSystem, OrderedIndexHandler, SteadyState,
    DefaultIndexHandler, NaiveIndexHandler

# Reexported model-definition and problem interface; approved via `reexports_allow`
# in test/qa/qa.jl and documented in docs/src/mainapi.md.
export @reaction_network, ReactionSystem, numspecies, reactions, species
export ODEFunction, ODEProblem, SteadyStateProblem, solve

_unresolve1(x) = x
_unresolve1(f::Function) = nameof(f)

_prewalk(f, x) = _prewalk_children(f(x), f)
_prewalk_children(x, f) = x
_prewalk_children(x::Expr, f) = Expr(x.head, map(arg -> _prewalk(f, arg), x.args)...)

function _striplines(ex)
    ex isa Expr || return ex
    if ex.head === :macrocall
        # A macro call requires its LineNumberNode in the second slot.
        return Expr(:macrocall, ex.args[1:2]..., _striplines.(ex.args[3:end])...)
    end
    args = (arg for arg in ex.args if !(arg isa LineNumberNode))
    return Expr(ex.head, _striplines.(args)...)
end

function _flatten(ex)
    ex isa Expr || return ex
    args = _flatten.(ex.args)
    ex.head === :block || return Expr(ex.head, args...)

    flattened = Any[]
    for arg in args
        arg isa Expr && arg.head === :block ? append!(flattened, arg.args) : push!(flattened, arg)
    end
    return length(flattened) == 1 ? only(flattened) : Expr(:block, flattened...)
end

_unresolve(ex) = _prewalk(_unresolve1, ex)
function _prettify(ex; lines = false)
    ex = lines ? ex : _striplines(ex)
    return _unresolve(_flatten(ex))
end

"""
    AbstractIndexHandler

Abstract supertype for index handlers used by [`FSPSystem`](@ref).

# Extension Rules
To implement a custom handler, subtype `AbstractIndexHandler` and extend the
documented index-handler interface described in the manual. Matrix-capable handlers
must implement both forms of `singleindices` and `pairedindices`, `LinearIndices`, and
`vec`; all handlers must implement `getsubstitutions`.
"""
abstract type AbstractIndexHandler end

include("fspsystem.jl")
include("indexhandlers.jl")
include("deprecated.jl")
include("build_rhs.jl")
include("build_rhs_ss.jl")
include("matrix.jl")

@setup_workload begin
    @compile_workload begin
        reaction_system = Catalyst.@reaction_network begin
            birth, 0 --> A
            death, A --> 0
        end
        system = FSPSystem(reaction_system)
        SparseArrays.SparseMatrixCSC(system, (3,), [:birth => 1.0, :death => 0.5], 0)
    end
end

end
