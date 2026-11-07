# Internal API

```@meta
CurrentModule = FiniteStateProjection
```

## [Index Handlers](@id index_handlers_internal)

### Index Handler Implementations

User-defined index handlers should inherit from `AbstractIndexHandler` and implement the following methods:

  - [`getsubstitutions`](@ref getsubstitutions)
  - [`build_rhs_header`](@ref build_rhs_header)
  - [`singleindices`](@ref singleindices)
  - [`pairedindices`](@ref pairedindices)

For matrix conversions they should additionally implement:

  - [`LinearIndices`](@ref Base.LinearIndices)
  - [`vec`](@ref Base.vec)

### Built-in implementations

```@docs
getsubstitutions(::OrderedIndexHandler, ::ReactionSystem)
```

## Function Building

```@docs
build_rhs
unpackparams
build_ratefuncs
build_rhs_firstpass
build_rhs_secondpass
```

## Steady-State Functions

```@docs
build_rhs_ss
build_rhs_singlepass_ss
```
