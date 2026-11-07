"""
    FSPSystem(rs::Catalyst.ReactionSystem, [ih]; combinatoric_ratelaw = true)

Represent a Catalyst reaction system as a finite state projection (FSP) system.
`FSPSystem` stores the reaction system, an index handler describing the state-array
layout, and generated rate functions used to construct ODE and steady-state problems.

# Fields
- `rs`: Underlying Catalyst reaction system.
- `ih`: Index handler defining the FSP state-array layout.
- `rfs`: Generated reaction-rate functions in the index-handler representation.

# Arguments
- `rs`: Catalyst reaction system without subsystems.
- `ih`: Index handler for the FSP state array, which determines how species are laid out in state space.
        [`OrderedIndexHandler`](@ref) assigns each species a separate axis in a user-defined order. 
        The default order is that returned by `Catalyst.species`.

# Keyword Arguments
- `combinatoric_ratelaw`: Whether to use combinatoric jump rate laws.

# Examples
```julia
using Catalyst

rn = @reaction_network begin
    birth, 0 --> A
    death, A --> 0
end
fsp = FSPSystem(rn)
```
"""
struct FSPSystem{IHT <: AbstractIndexHandler, RT}
    rs::ReactionSystem
    ih::IHT
    rfs::RT
end

struct RateFunction{F, E}
    callable::F
    expression::E
end

@inline (rf::RateFunction)(args...) = rf.callable(args...)

function FSPSystem(
        rs::ReactionSystem,
        ih::AbstractIndexHandler = OrderedIndexHandler(rs);
        combinatoric_ratelaw::Bool = true
    )
    isempty(get_systems(rs)) ||
        error("Supported Catalyst models can not contain subsystems. Use `ModelingToolkitBase.flatten(rs)` to generate a single system with no subsystems from your Catalyst model.")
    any(eq -> !(eq isa Reaction), equations(rs)) &&
        error("Catalyst models that include constraint ODEs or algebraic equations are not supported.")

    rfs = create_ratefuncs(rs, ih; combinatoric_ratelaw = combinatoric_ratelaw)
    return FSPSystem(rs, ih, rfs)
end

function FSPSystem(rs::ReactionSystem, order::AbstractVector; offset = 1, kwargs...)
    return FSPSystem(rs, OrderedIndexHandler(rs, order; offset); kwargs...)
end

"""
    build_ratefuncs(rs, ih; state_sym::Symbol, combinatoric_ratelaw::Bool)::Vector

Return the rate functions converted to Julia expressions in the state variable
`state_sym`. Abundances of the species are computed using `getsubstitutions`.

See also: [`getsubstitutions`](@ref), [`build_rhs`](@ref)
"""
function build_ratefuncs(
        rs::ReactionSystem, ih::AbstractIndexHandler;
        state_sym::Symbol, combinatoric_ratelaw::Bool = true
    )
    nspecs = numspecies(rs)
    state = (@variables ($state_sym)[1:nspecs])[1]
    @variables t
    params = parameters(rs)
    substitutions = getsubstitutions(ih, rs, state_sym = state_sym)

    return map(reactions(rs)) do reac
        jrl = jumpratelaw(reac; combinatoric_ratelaw)
        rate = substitute(jrl, substitutions)
        ex = build_function(rate, state, t, params...; expression = Val{true})
        ex isa Expr && ex.head === :function && length(ex.args) == 2 ||
            throw(ArgumentError("Symbolics.build_function returned an unsupported expression"))
        ex.args[2]
    end
end

function create_ratefuncs(rs::ReactionSystem, ih::AbstractIndexHandler; combinatoric_ratelaw::Bool = true)
    params = getname.(parameters(rs))

    return tuple(
        map(
            rate -> compile_ratefunc(rate, params),
            build_ratefuncs(rs, ih; state_sym = :idx_in, combinatoric_ratelaw)
        )...
    )
end

function compile_ratefunc(rate, params)
    ex = _flatten(:((idx_in, t, $(params...)) -> $(rate)))
    return RateFunction(@RuntimeGeneratedFunction(ex), rate)
end

_parameter_symbol(key::Symbol) = key
_parameter_symbol(key) = getname(key)

function pmap_to_p(sys::FSPSystem, pmap)
    pmap isa SciMLBase.NullParameters && return pmap
    values_by_parameter = Dict(_parameter_symbol(k) => v for (k, v) in pmap)
    return [values_by_parameter[p] for p in getname.(parameters(sys.rs))]
end
