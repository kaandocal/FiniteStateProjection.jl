module AxisArraysExt

using AxisArrays: AxisArray, axisnames, axes
using Catalyst: ReactionSystem, species
using FiniteStateProjection: FSPSystem, OrderedIndexHandler
import SciMLBase
using SciMLBase: ODEProblem, SteadyStateProblem
using SymbolicIndexingInterface: getname

# Names of the species in the order of the state-array dimensions.
function _speciesnames(sys::FSPSystem{<:OrderedIndexHandler})
    spec = species(sys.rs)
    return [getname(spec[i]) for i in sys.ih.perm]
end

"""
    FSPSystem(rs::ReactionSystem, axisarray::AxisArray; offset = nothing, kwargs...)

Return an `FSPSystem` whose state layout comes from `axisarray`. Each axis
must match a species of `rs`, and the species order follows the axis order.
Offsets can be passed as a list of pairs (e.g. `[:A => 2]`), and default to 1
unless otherwise specified.

Requires AxisArrays.jl to be loaded.

# Examples
```julia
using AxisArrays

axisarray = AxisArray(zeros(51, 51), Axis{:B}(1:50), Axis{:A}(2:52))
sys = FSPSystem(rn, axisarray; offset = [:A => 2])  # B has offset 1, A has offset 2
```
"""
function _system_from_template(rs::ReactionSystem, template::AxisArray, offset; kwargs...)
    order = collect(axisnames(template))
    offsets = fill(1, length(order))
    if !isnothing(offset)
        for (name, value) in offset
            i = findfirst(==(Symbol(name)), order)
            isnothing(i) && throw(
                ArgumentError("Cannot set the offset of $name, which is not an axis of the template")
            )
            value isa Integer || throw(ArgumentError("The offset of $name must be an integer, got $value"))
            offsets[i] = Int(value)
        end
    end
    return FSPSystem(rs, OrderedIndexHandler(rs, order; offset = offsets); kwargs...)
end

function (::Type{FSPSystem})(rs::ReactionSystem, template::AxisArray; offset = nothing, kwargs...)
    return _system_from_template(rs, template, offset; kwargs...)
end

# A one-dimensional AxisArray is also an AbstractVector, so this method resolves the
# ambiguity with the vector-of-species constructor.
function (::Type{FSPSystem})(rs::ReactionSystem, template::AxisArray{<:Any, 1}; offset = nothing, kwargs...)
    return _system_from_template(rs, template, offset; kwargs...)
end

# Converts a state array whose axes are labelled by species names into the plain array
# expected by the FSP system. Axis labels are matched by name, so the axes may be given
# in any order. Axis values are not checked: they only serve as labels.
function _reorder_state(sys::FSPSystem, u::AxisArray)
    target = _speciesnames(sys)
    names = collect(axisnames(u))

    length(names) == length(target) && Set(names) == Set(target) || throw(
        ArgumentError(
            "State array axes $(names) do not match the species $(target) of the FSP system"
        )
    )

    perm = [findfirst(==(name), names) for name in target]
    return permutedims(Base.parent(u), perm)
end

"""
    SciMLBase.ODEProblem(sys::FSPSystem, u0::AxisArray, tmax, p)

Return an `ODEProblem` for an initial distribution labelled by species. The axes of
`u0` must correspond to the species of `sys`, in any order. 

Requires AxisArrays.jl to be loaded.

# Examples
```julia
using AxisArrays

u0 = AxisArray(zeros(51, 51), Axis{:B}(0:50), Axis{:A}(0:50))
u0[1, 1] = 1.0
prob = ODEProblem(sys, u0, (0.0, 10.0), pmap)
```
"""
function SciMLBase.ODEProblem(sys::FSPSystem, u0::AxisArray, tint, pmap = SciMLBase.NullParameters())
    return ODEProblem(sys, _reorder_state(sys, u0), tint, pmap)
end

"""
    SciMLBase.SteadyStateProblem(sys::FSPSystem, u0::AxisArray, p)

Return a `SteadyStateProblem` for an initial distribution labelled by species. See
[`SciMLBase.ODEProblem`](@ref) for the axis rules. 
"""
function SciMLBase.SteadyStateProblem(sys::FSPSystem, u0::AxisArray, pmap = SciMLBase.NullParameters())
    return SteadyStateProblem(sys, _reorder_state(sys, u0), pmap)
end

end
