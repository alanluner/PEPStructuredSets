## code to compute projection onto a psd cone

#####-----------------------------#####
# This routine comes from the Branch-And-Bound PEP solver of Shuvomoy Das Gupta, Bart P.G. Van Parys, and Ernest K. Ryu.
# Accessible at https://github.com/Shuvomoy/BnB-PEP-code
# 
# Shuvomoy Das Gupta, Bart P.G. Van Parys, Ernest K. Ryu, "Branch-and-Bound Performance Estimation Programming: A Unified Methodology for Constructing Optimal Optimization Methods",
# Mathematical Programming 204.1 (2024): 567-639.
#####-----------------------------#####

# The following part is taken from ProximalOperators.jl

#  ProximalOperators.jl code start

const RealOrComplex{R <: Real} = Union{R, Complex{R}}
const HermOrSym{T, S} = Union{Hermitian{T, S}, Symmetric{T, S}}

abstract type ProximableFunction end


struct IndPSD <: ProximableFunction
    scaling::Bool
end

IndPSD(; scaling=false) = IndPSD(scaling)

function (f::IndPSD)(X::HermOrSym{T}) where {R <: Real, T <: RealOrComplex{R}}
    F = eigen(X)
    for i in eachindex(F.values)
        # Do we allow for some tolerance here?
        if F.values[i] <= -100 * eps(R)
            return R(Inf)
        end
    end
    return R(0)
end


function (f::IndPSD)(X::AbstractMatrix{R}) where R <: Real
    f(Symmetric(X))
end

function (f::IndPSD)(X::AbstractMatrix{C}) where C <: Complex
    f(Hermitian(X))
end

is_convex(f::IndPSD) = true
is_cone(f::IndPSD) = true

fun_name(f::IndPSD) = "indicator of positive semidefinite cone"
fun_dom(f::IndPSD) = "Symmetric, Hermitian, AbstractArray{Float64}"
fun_expr(f::IndPSD) = "x ↦ 0 if A ⪰ 0, +∞ otherwise"
fun_params(f::IndPSD) = "none"

function projection_on_psdcone!(Y, f::IndPSD, X; gamma=1.0, ϵ_min_pos_eig_val = 1e-10)
    n = size(X, 1)
    F = eigen(X)
    for i in eachindex(F.values)
        F.values[i] = max.(ϵ_min_pos_eig_val, F.values[i])
    end
    for i = 1:n, j = i:n
        Y[i, j] = 0.0
        for k = 1:n
            Y[i, j] += F.vectors[i, k] * F.values[k] * conj(F.vectors[j, k])
        end
        Y[j, i] = conj(Y[i, j])
    end
    return 0.0
end

#  ProximalOperators.jl code end

## function to compute the pivoted choleksy decomposition of a positive semidefinite matrix

function compute_pivoted_cholesky_L_mat(A; ϵ_tol = 1e-4, suppressWarning = false)
    # ensure that A is positive semidefinite
    Y = zeros(size(A))
    n = size(A, 1)
    # Find the psd projection of A
    projection_on_psdcone!(Y, IndPSD(), A)
    if (norm(Y-A) >= 1e-6)&&(!suppressWarning)
        @warn "the given matrix is not positive semidefinite (no need to panic if you are computing a lower bound)"
    end
    F = cholesky(Y; check=false)
    F_L_actual = F.L
    for i in 1:n
        for j in 1:n
            if i >= j
                if abs(F_L_actual[i,j]) <= ϵ_tol
                    F_L_actual[i,j] = 0.0
                end
            end
        end
    end
    # F = cholesky(Y, Val(true); check=false)
    # F_L = F.L
    # F_P = F.P # permutation matrix
    # F_L_actual = F_P*F_L # because we have A = (F.P*F.L)*(F.P*F.L)'
    # for i in 1:n
    #     for j in 1:n
    #         if i < j
    #             F_L_actual[i,j] = 0.0
    #         end
    #     end
    # end
    return F_L_actual
end