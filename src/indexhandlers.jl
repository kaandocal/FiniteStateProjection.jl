"""
    singleindices(idxhandler::AbstractIndexHandler, arr)

Returns the indices of the finite state space represented by `arr`.

# Arguments
- `idxhandler`: The index handler that defines the state-space layout.
- `arr`: Either the state array or its tuple of dimensions.

# Returns
- An iterator of Cartesian indices. The default implementation returns
  `CartesianIndices(arr)`.

# Extension Rules
Custom index handlers must import and extend this function. Implement both the
state-array and dimension-tuple forms when supporting matrix conversions.
"""
singleindices(::AbstractIndexHandler, arr::AbstractArray) = CartesianIndices(arr)
singleindices(::AbstractIndexHandler, arr::Tuple) = CartesianIndices(arr)

"""
    pairedindices(idxhandler::AbstractIndexHandler, arr, shift::CartesianIndex)

Returns pairs `(I .- shift, I)` for state transitions that remain within `arr`.

# Arguments
- `idxhandler`: The index handler that defines the state-space layout.
- `arr`: Either the state array or its tuple of dimensions.
- `shift`: The stoichiometry vector for the transition.

# Returns
- An iterator of source and destination Cartesian-index pairs.

# Extension Rules
Custom index handlers must import and extend this function. Implement both the
state-array and dimension-tuple forms when supporting matrix conversions.
"""
function pairedindices end

"""
    getsubstitutions(idxhandler::AbstractIndexHandler, rs::ReactionSystem; state_sym::Symbol)

Returns a dictionary `S_i => f_i(state_sym)`, where each `f_i` computes a species
abundance from the symbolic state variable.

# Arguments
- `idxhandler`: The index handler that defines the state-space layout.
- `rs`: Catalyst reaction system whose species require substitutions.

# Keyword Arguments
- `state_sym`: Symbol naming the generated RHS state variable.

# Returns
- A dictionary mapping each Catalyst species to a symbolic abundance expression.

# Extension Rules
Custom index handlers must import and extend this function.
"""
function getsubstitutions end

"""
    vec(idxhandler::AbstractIndexHandler, arr)

Converts the CME state array into the vector ordering used by sparse-matrix
representations.

# Arguments
- `idxhandler`: The index handler that defines the vector ordering.
- `arr`: State array to flatten.

# Returns
- A one-dimensional state vector compatible with [`LinearIndices`](@ref Base.LinearIndices).

# Extension Rules
Custom index handlers that support matrix conversions must import and extend this
`Base` function.

See also: [`LinearIndices`](@ref Base.LinearIndices)
"""
function vec end

"""
    LinearIndices(idxhandler::AbstractIndexHandler, arr)

Returns an object `lind` converting the Cartesian indices produced by
[`singleindices`](@ref) and [`pairedindices`](@ref) to the linear indices of
[`vec`](@ref Base.vec). The required invariant is

```julia
arr[idx_cart] == vec(idxhandler, arr)[idx_lin]
```

# Arguments
- `idxhandler`: The index handler that defines the vector ordering.
- `arr`: Either the state array or its tuple of dimensions.

# Returns
- An indexable object mapping Cartesian indices to vector positions.

# Extension Rules
Custom index handlers that support matrix conversions must import and extend this
`Base` function, preserving the documented indexing invariant.

See also: [`vec`](@ref Base.vec)
"""
function LinearIndices end

##

"""
    OrderedIndexHandler(rs::ReactionSystem; offset = 1)
    OrderedIndexHandler(rs::ReactionSystem, order; offset = 1)

Index handler for an FSP system with `N` species. Each species gets its own axis of the
state array, in the order `order` (defaults to the order returned by `Catalyst.species`). 
Dimension `i` of the state array tracks the species `order[i]`.

# Arguments
- `rs`: Catalyst reaction system.
- `order`: The species of `rs`, given as species names (`Symbol`s) or Catalyst species.
  Every species must appear exactly once. By default, `order` follows `Catalyst.species`.

# Keyword Arguments
- `offset`: Offset for each species (index corresponding to molecule counts of 0). 
  Can be a single number, which is used for all species. Defaults to `1` in agreement with Julia's one-based indexing. 

This representation is appropriate when every state in the truncated array is reachable;
reduce conserved species before construction when possible.

# Examples
```julia
rn = @reaction_network begin
    birth, 0 --> A
    death, A --> B
end
ih = OrderedIndexHandler(rn, [:B, :A]; offset = 1)  # dimension 1 tracks B, dimension 2 tracks A
```
"""
struct OrderedIndexHandler{N} <: AbstractIndexHandler
    offset::NTuple{N, Int}
    perm::NTuple{N, Int}
end

# Expands an offset given as one integer, or as one integer per dimension.
_offsets(offset::Integer, N::Integer) = ntuple(_ -> Int(offset), N)

function _offsets(offset, N::Integer)
    length(offset) == N || throw(
        DimensionMismatch(
            "Length of offset ($(length(offset))) does not match number of species ($N)"
        )
    )
    return ntuple(i -> Int(offset[i]), N)
end

function OrderedIndexHandler(rs::ReactionSystem; offset = 1)
    N = numspecies(rs)
    return OrderedIndexHandler{N}(_offsets(offset, N), Tuple(1:N))
end

# Position of `sp` among the species of the reaction system `spec`.
function _species_position(spec, sp::Symbol)
    index = findfirst(s -> getname(s) == sp, spec)
    isnothing(index) && throw(ArgumentError("Cannot find species $sp in reaction system"))
    return index
end

function _species_position(spec, sp)
    index = findfirst(s -> isequal(s, sp), spec)
    isnothing(index) && throw(ArgumentError("Cannot find species $sp in reaction system"))
    return index
end

function OrderedIndexHandler(rs::ReactionSystem, order::AbstractVector; offset = 1)
    spec = species(rs)
    nspec = length(spec)

    length(order) == nspec || throw(
        DimensionMismatch(
            "Length of species vector ($(length(order))) does not match number of species ($nspec)"
        )
    )

    perm = zeros(Int, nspec)
    seen = falses(nspec)
    for (i, sp) in enumerate(order)
        index = _species_position(spec, sp)
        seen[index] && throw(ArgumentError("Species $sp specified twice in ordering"))
        seen[index] = true
        perm[i] = index
    end

    return OrderedIndexHandler{nspec}(_offsets(offset, nspec), Tuple(perm))
end

Base.vec(::OrderedIndexHandler, arr) = vec(arr)

# Maps state values to positions in `vec(arr)`. The state values run over the axes of the
# array, which may start anywhere, while the positions run from 1 in column-major order.
struct ShiftedLinearIndices{N, L <: LinearIndices{N}}
    positions::L
    firsts::NTuple{N, Int}
end

Base.length(lind::ShiftedLinearIndices) = length(lind.positions)

function Base.getindex(lind::ShiftedLinearIndices, idx::CartesianIndex)
    return lind.positions[CartesianIndex(map((i, f) -> i - f + 1, Tuple(idx), lind.firsts))]
end

_axis_length(ax::Integer) = ax
_axis_length(ax) = length(ax)
_axis_first(ax::Integer) = 1
_axis_first(ax) = Int(first(ax))

function Base.LinearIndices(::OrderedIndexHandler, dims::Tuple)
    shape = map(_axis_length, dims)
    firsts = map(_axis_first, dims)
    return ShiftedLinearIndices(LinearIndices(shape), firsts)
end

Base.LinearIndices(ih::OrderedIndexHandler, arr::AbstractArray) = LinearIndices(ih, axes(arr))

function pairedindices(
        ih::OrderedIndexHandler{N}, arr::AbstractArray{T, N},
        shift::CartesianIndex{N}
    ) where {T, N}
    return pairedindices(ih, axes(arr), shift)
end

# `dims` is written as `Tuple{T, Vararg{T}}` rather than `NTuple{N, T}` so that the
# element type `T` is always bound: `NTuple{0, T} === Tuple{}` leaves `T` free, which
# trips Aqua's unbound-type-parameter check. The zero-dimensional case is handled by the
# explicit method below.
function pairedindices(
        ih::OrderedIndexHandler{N}, dims::Tuple{T, Vararg{T}},
        shift::CartesianIndex{N}
    ) where {N, T <: Number}
    length(dims) == N || throw(
        DimensionMismatch(
            "Dimension of state space ($(length(dims))) does not match number of species ($N)"
        )
    )
    return pairedindices(ih, Base.OneTo.(dims), shift)
end

# Handles the degenerate zero-dimensional case (no species), where neither `T`-constrained
# method below matches because `Tuple{}` has no element type.
function pairedindices(
        ::OrderedIndexHandler{0}, ::Tuple{}, ::CartesianIndex{0}
    )
    return zip(CartesianIndices(()), CartesianIndices(()))
end

# Important: the species in `shift` are ordered according to `Catalyst.species`!
function pairedindices(
        ih::OrderedIndexHandler{N}, dims::Tuple{T, Vararg{T}},
        shift::CartesianIndex{N}
    ) where {N, T <: AbstractVector}
    length(dims) == N || throw(
        DimensionMismatch(
            "Dimension of state space ($(length(dims))) does not match number of species ($N)"
        )
    )
    ranges = tuple(
        (
            UnitRange(
                max(first(ax), first(ax) + shift[ih.perm[i]]),
                min(last(ax), last(ax) + shift[ih.perm[i]])
            )
                for (i, ax) in enumerate(dims)
        )...
    )

    ranges_shifted = tuple((rng .- shift[ih.perm[i]] for (i, rng) in enumerate(ranges))...)

    return zip(CartesianIndices(ranges_shifted), CartesianIndices(ranges))
end

function pairedindices(
        ::OrderedIndexHandler, dims::Tuple,
        shift::CartesianIndex
    )
    throw(
        DimensionMismatch(
            "Dimension of state space ($(length(dims))) does not match number of species ($(length(shift)))"
        )
    )
end

"""
    getsubstitutions(ih::OrderedIndexHandler, rs::ReactionSystem; state_sym::Symbol)::Dict

Defines the abundance of species ``S_i`` to be `state_sym[i] - offset[i]`, where dimension
`i` of the state array tracks the species `perm[i]` of `rs`.
"""
function getsubstitutions(ih::OrderedIndexHandler, rs::ReactionSystem; state_sym::Symbol)
    nspecs = numspecies(rs)
    state_array = (@variables ($state_sym)[1:nspecs])[1]
    state_sym_vec = [state_array[i] for i in 1:nspecs]

    species_orig = species(rs)
    species_perm = [species_orig[ih.perm[i]] for i in 1:nspecs]

    return Dict(symbol => state_sym_vec[i] - ih.offset[i] for (i, symbol) in enumerate(species_perm))
end
