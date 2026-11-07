# Deprecated index-handler names, kept so that existing code keeps working. They are
# scheduled for removal in the next breaking release.

"""
    DefaultIndexHandler{N}(offset, perm)

Deprecated alias of [`OrderedIndexHandler`](@ref). Use `OrderedIndexHandler` instead.
"""
const DefaultIndexHandler = OrderedIndexHandler

# The positional `(offset, perm)` form took a single offset. It is kept only for the
# deprecated names above.
OrderedIndexHandler{N}(offset::Integer, perm::NTuple{N, Int}) where {N} =
    OrderedIndexHandler{N}(_offsets(offset, N), perm)
OrderedIndexHandler(offset::Integer, perm::NTuple{N, Int}) where {N} =
    OrderedIndexHandler{N}(_offsets(offset, N), perm)

"""
    NaiveIndexHandler

Deprecated constructor forwarding to [`OrderedIndexHandler`](@ref).

# Arguments
- `args...`: Positional arguments accepted by `OrderedIndexHandler`.

# Keyword Arguments
- `kwargs...`: Keyword arguments accepted by `OrderedIndexHandler`.

Use `OrderedIndexHandler` directly instead.
"""
function NaiveIndexHandler(args...; kwargs...)
    Base.depwarn(
        "`NaiveIndexHandler` is deprecated, use `OrderedIndexHandler` instead.",
        :NaiveIndexHandler
    )
    return OrderedIndexHandler(args...; kwargs...)
end

"""
    PermutingIndexHandler(rs::ReactionSystem, order)

Deprecated constructor forwarding to [`OrderedIndexHandler`](@ref)`(rs, order)`.
"""
function PermutingIndexHandler(rs::ReactionSystem, order::AbstractVector)
    Base.depwarn(
        "`PermutingIndexHandler` is deprecated, use `OrderedIndexHandler(rs, order)` instead.",
        :PermutingIndexHandler
    )
    return OrderedIndexHandler(rs, order)
end
