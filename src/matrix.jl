function create_sparsematrix(sys::FSPSystem, dims::NTuple, ps, t)
    lind = LinearIndices(sys.ih, dims)
    Ntot = length(lind)

    I = Int[]
    J = Int[]
    V = Float64[]

    predsize = Ntot * (length(equations(sys.rs)) + 1)

    sizehint!(I, predsize)
    sizehint!(J, predsize)
    sizehint!(V, predsize)

    for idx_cart in singleindices(sys.ih, dims)
        idx_lin = lind[idx_cart]
        push!(I, idx_lin)
        push!(J, idx_lin)

        rate = 0.0
        for rf in sys.rfs
            rate -= rf(idx_cart, t, ps...)
        end

        push!(V, rate)
    end

    S::Matrix{Int64} = netstoichmat(sys.rs)
    for (i, rf) in enumerate(sys.rfs)
        for (idx_cin, idx_cout) in pairedindices(sys.ih, dims, CartesianIndex(S[:, i]...))
            idx_lin = lind[idx_cin]
            idx_lout = lind[idx_cout]
            push!(I, lind[idx_cout])
            push!(J, lind[idx_cin])

            rate = rf(idx_cin, t, ps...)
            push!(V, rate)
        end
    end

    return sparse(I, J, V, Ntot, Ntot)
end

function create_sparsematrix_ss(sys::FSPSystem, dims::NTuple, ps)
    lind = LinearIndices(sys.ih, dims)
    Ntot = length(lind)

    I = Int[]
    J = Int[]
    V = Float64[]

    predsize = 2 * Ntot * length(equations(sys.rs))

    sizehint!(I, predsize)
    sizehint!(J, predsize)
    sizehint!(V, predsize)

    S::Matrix{Int64} = netstoichmat(sys.rs)
    for (i, rf) in enumerate(sys.rfs)
        for (idx_cin, idx_cout) in pairedindices(sys.ih, dims, CartesianIndex(S[:, i]...))
            idx_lin = lind[idx_cin]
            idx_lout = lind[idx_cout]
            push!(I, lind[idx_cout])
            push!(J, lind[idx_cin])

            rate = rf(idx_cin, 0, ps...)
            push!(V, rate)

            push!(I, lind[idx_cin])
            push!(J, lind[idx_cin])
            push!(V, -rate)
        end
    end

    return sparse(I, J, V, Ntot, Ntot)
end

"""
    SparseArrays.SparseMatrixCSC(sys::FSPSystem, dims::NTuple, ps, t::Real)

Converts an FSP system into a sparse matrix defining the time-dependent Chemical
Master Equation.

# Arguments
- `sys`: FSP system to convert.
- `dims`: Dimensions of the truncated FSP state array.
- `pmap`: Iterable of parameter-value pairs, or `SciMLBase.NullParameters()`.
- `t`: Time at which to evaluate a time-dependent rate law.

# Returns
- A sparse matrix acting on the state vector produced by [`vec`](@ref Base.vec).
"""
function SparseArrays.SparseMatrixCSC(sys::FSPSystem, dims::NTuple, pmap, t::Real)
    return create_sparsematrix(sys, dims, pmap_to_p(sys, pmap), t)
end

"""
    SparseArrays.SparseMatrixCSC(sys::FSPSystem, dims::NTuple, ps, ::SteadyState)

Converts an FSP system into the sparse matrix for its steady-state Chemical Master
Equation formulation.

# Arguments
- `sys`: FSP system to convert.
- `dims`: Dimensions of the truncated FSP state array.
- `pmap`: Iterable of parameter-value pairs, or `SciMLBase.NullParameters()`.
- `SteadyState()`: Dispatch marker selecting the formulation that drops transitions
  leaving the truncated state space.

# Returns
- A sparse matrix acting on the state vector produced by [`vec`](@ref Base.vec).
"""
function SparseArrays.SparseMatrixCSC(sys::FSPSystem, dims::NTuple, pmap, ::SteadyState)
    return create_sparsematrix_ss(sys, dims, pmap_to_p(sys, pmap))
end
