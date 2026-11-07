module AxisArraysExt

using AxisArrays: AxisArray, axisnames
using Catalyst: ReactionSystem
using FiniteStateProjection: FSPSystem, OrderedIndexHandler

"""
    FSPSystem(rs::ReactionSystem, axisarray::AxisArray; offset = nothing, kwargs...)

Return an `FSPSystem` whose state layout comes from `axisarray`. Each axis
must match a species of `rs`, and the species order follows the axis order.
Offsets can be passed as a list of pairs (e.g. `[:A => 2]`), and default to 1
unless otherwise specified.

Problems built from a state array labelled with the same axes, for example
`ODEProblem(sys, u0)` with `u0` an `AxisArray`, return solutions that are labelled
in the same way.

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

end
