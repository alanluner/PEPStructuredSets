include("FW_ConvexStepsizeOptimization.jl")
include("code_to_compute_pivoted_cholesky.jl")


#####-----------------------------#####
# This code is adapted from the Branch-And-Bound PEP solver of Shuvomoy Das Gupta, Bart P.G. Van Parys, and Ernest K. Ryu.
# Accessible at https://github.com/Shuvomoy/BnB-PEP-code
# 
# Shuvomoy Das Gupta, Bart P.G. Van Parys, Ernest K. Ryu, "Branch-and-Bound Performance Estimation Programming: A Unified Methodology for Constructing Optimal Optimization Methods",
# Mathematical Programming 204.1 (2024): 567-639.
#####-----------------------------#####

# Solve dual for p_Relaxed for FW over smooth sets
function solve_dual_FW_SmoothRelaxation(N, h, R, L, μ, β; functionType = :smooth, optLoc = :exterior, primalObjective = :minIterate, dualObjective = :default, obj_val_upper_bound = 1e6, show_output = :off, print_model = :off)

    # data generator
    # --------------
    𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetSmooth(N, h)

    if optLoc!=:exterior
        @warn "Dual is not functional for optLoc :inSet or :interior"
    end
    

    # Number of points etc
    # --------------------

    I = -1:N-1
    K = -1:N
    dim_Z = 4N+6

    # define the model
    # ----------------

    model_dual = Model(optimizer_with_attributes(Mosek.Optimizer))
    # set_optimizer_attribute(model_dual, "MSK_DPAR_INTPNT_TOL_PFEAS", 1.0e-8)

    # define the variables
    # --------------------

    # define the index sets of dual variables
    idx_set_λ = index_set_constructor(K)        # Constraint: f Smooth convex
    idx_set_η = index_set_constructor(I)        # Constraint: Smooth convex
    idx_set_ϕ = index_set_constructor(I,K)      # Constraint: w ∈ Set
    idx_set_κ = index_set_constructor_symm(K) # Constraint: w-w diameter
    idx_set_ν = index_set_constructor(I,K)   # Constraint: w-z_shift diameter
    idx_set_ρ = index_set_constructor_symm(I)   # Constraint: z_shift-z_shift diameter
    idx_set_ω = index_set_constructor_single(K)   # Constraint: x-w link
    idx_set_σ = index_set_constructor_single(I)     # Constraint: gi, ni alignment
    idx_set_χ = index_set_constructor(I)    # Constraint: g-ni_nj relaxation
    idx_set_ζ = index_set_constructor_single(I)     #Constraint: n normalization


    # define dual vars
    @variable(model_dual, λ[idx_set_λ] >= 0)
    @variable(model_dual, η[idx_set_η] >= 0)
    @variable(model_dual, ϕ[idx_set_ϕ] >= 0)
    @variable(model_dual, κ[idx_set_κ] >= 0)
    @variable(model_dual, ν[idx_set_ν] >= 0)
    @variable(model_dual, ρ[idx_set_ρ] >= 0)
    @variable(model_dual, ω[idx_set_ω] >= 0)
    @variable(model_dual, σ[idx_set_σ] >= 0)
    @variable(model_dual, χ[idx_set_χ] >= 0)
    @variable(model_dual, ζ[idx_set_ζ]) # We shouldn't require >=0, since it is for an equality condition

    if primalObjective == :minIterate
        # if objective is minIterate, then introduce additional variable τ
        @variable(model_dual, τ[1:N+1] >= 0)
    else
        τ = 0
    end

    if optLoc == :inSet
        # if optLoc is inSet, then introduce additional variable ξ
        @variable(model_dual, ξ >= 0)
    elseif optLoc == :interior
        @warn "optLoc :interior not supported"
    else
        ξ = 0
    end

    # define Z ⪰ 0
    @variable(model_dual, Z[1:dim_Z, 1:dim_Z])
    @constraint(model_dual, psd, Z >=0, PSDCone())

    # group dual variables and index sets to make more compact
    dualVars = (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z)
    idx_sets = (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ)

    # set objective
    setDualObjectiveSmoothRelaxationFW(model_dual, R, β, dualVars, idx_sets; dualObjective = dualObjective, obj_val_upper_bound = obj_val_upper_bound)

    # add the linear constraint
    # -------------------------

    addDualLinearConstraintFW_SmoothRelaxation(model_dual, N, primalObjective, λ, idx_set_λ, τ, 𝐟)

    # add the LMI constraint
    # ----------------------

    addDualLMIConstraintSmoothRelaxationFW(model_dual, functionType, optLoc, β, L, μ, 𝐠, 𝐱, 𝐳, 𝐧, 𝐰, dualVars, idx_sets)


    # optimize
    # ----------------

    if show_output == :off
        set_silent(model_dual)
    end

    optimize!(model_dual)

    if termination_status(model_dual) != MOI.OPTIMAL
		#  @info "💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀💀"
        # @error "model_dual solving did not reach optimality;  termination status = " termination_status(model_dual)
    end

    # store the solutions and return
    # ------------------------------

    # store λ_opt

    λ_opt = value.(λ)
    η_opt = value.(η)
    ϕ_opt = value.(ϕ)
    κ_opt = value.(κ)
    ν_opt = value.(ν)
    ρ_opt = value.(ρ)
    ω_opt = value.(ω)
    σ_opt = value.(σ)
    χ_opt = value.(χ)
    ζ_opt = value.(ζ)

    if primalObjective == :minIterate
        τ_opt = value.(τ)
    else
        τ_opt = 0
    end

    if optLoc in [:inSet, :interior]
        ξ_opt = value.(ξ)
    else
        ξ_opt = 0
    end

    # store Z_opt

    Z_opt = value.(Z)

    # compute cholesky

    Λ_opt =  compute_pivoted_cholesky_L_mat(Z_opt; ϵ_tol = 1e-6)

    if norm(Z_opt - Λ_opt*Λ_opt', Inf) > 1e-6
        # @info "checking the norm bound"
        # @warn "||Z - L*L^T|| = $(norm(Z_opt - Λ_opt*Λ_opt', Inf))"
    end

    #Only consider effective index for variables that aren't in objective
    idx_set_λ_opt_eff = effective_index_set_finder(λ_opt ; ϵ_tol = 1e-6)
    idx_set_η_opt_eff = effective_index_set_finder(η_opt ; ϵ_tol = 1e-6)
    idx_set_ϕ_opt_eff = effective_index_set_finder(ϕ_opt ; ϵ_tol = 1e-6)
    idx_set_σ_opt_eff = effective_index_set_finder_single(σ_opt ; ϵ_tol = 1e-6)
    idx_set_χ_opt_eff = effective_index_set_finder(χ_opt ; ϵ_tol = 1e-6)

    # return all the stored values

    if print_model == :on
        print(model_dual)
    end

    #DO NOT CALL objective_value here, because we might be using a different objective
    objVal = sum(ζ_opt[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω_opt[ij] for ij in idx_set_ω; init=0) + (R-2/β)^2*( sum(κ_opt[ij] for ij in idx_set_κ; init=0)  +  sum(ν_opt[ij] for ij in idx_set_ν; init=0)  +  sum(ρ_opt[ij] for ij in idx_set_ρ; init=0))

    # group output values for compact output
    dualVars_opt = (λ_opt, η_opt, ϕ_opt, κ_opt, ν_opt, ρ_opt, ω_opt, σ_opt, χ_opt, ζ_opt, τ_opt, ξ_opt, Z_opt, Λ_opt)
    idx_sets_eff = (idx_set_λ_opt_eff, idx_set_η_opt_eff, idx_set_ϕ_opt_eff, idx_set_σ_opt_eff, idx_set_χ_opt_eff)

    return objVal, dualVars_opt, h, idx_sets_eff, model_dual

end


function setDualObjectiveSmoothRelaxationFW(model, R, β, dualVars, idx_sets; dualObjective = :default, obj_val_upper_bound = 1e6)

    (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z) = dualVars
    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ) = idx_sets

    # Set objective to either the default dual objective, a sparsification objective, or an objective of maximizing one variable subject to an upper bound
    if dualObjective == :default
        @objective(model, Min, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)))
    elseif dualObjective == :find_sparse_sol
        #Minimize the variables that don't contribute to the objective function [λ, η, ϕ]
        @objective(model, Min, sum(λ[ij] for ij in idx_set_λ) + sum(η[ij] for ij in idx_set_η) + sum(ϕ[ij] for ij in idx_set_ϕ))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)

    elseif dualObjective == :find_M_Z
        @objective(model, Max, tr(Z))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)
    elseif dualObjective == :find_M_λ
        @objective(model, Max, sum(λ[ij] for ij in idx_set_λ))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)
    elseif dualObjective == :find_M_η
        @objective(model, Max, sum(η[ij] for ij in idx_set_η))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)
    elseif dualObjective == :find_M_ϕ
        @objective(model, Max, sum(ϕ[ij] for ij in idx_set_ϕ))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)
    elseif dualObjective == :find_M_χ
        @objective(model, Max, sum(χ[ij] for ij in idx_set_χ))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)
    elseif dualObjective == :find_M_σ
        @objective(model, Max, sum(σ[ij] for ij in idx_set_σ))
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (R-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= obj_val_upper_bound)
    else
        @warn "Invalid dualObjective"
    end
end

function addDualLinearConstraintFW_SmoothRelaxation(model, N, primalObjective, λ, idx_set_λ, τ, 𝐟)
    if primalObjective == :finalIterate
        @constraint(model,  sum(λ[ij_λ]*a_vec(ij_λ.i,ij_λ.j,𝐟) for ij_λ in idx_set_λ) - a_vec(-1,N,𝐟) .== 0)
    elseif primalObjective == :minIterate
        @constraint(model,  sum(λ[ij_λ]*a_vec(ij_λ.i,ij_λ.j,𝐟) for ij_λ in idx_set_λ) - sum(τ[i+1]*a_vec(-1,i,𝐟) for i in 0:N) .== 0) #We can reuse 𝐟 for τ since it has the same dimension
        @constraint(model, sum(τ) == 1)
    else
        @warn "Invalid primalObjective"
    end
end

function addDualLMIConstraintSmoothRelaxationFW(model, functionType, optLoc, β, L, μ, 𝐠, 𝐱, 𝐳, 𝐧, 𝐰, dualVars, idx_sets )

    (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z) = dualVars
    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ) = idx_sets

    if optLoc == :inSet
        if functionType == :smooth
            @constraint(model,
                sum(λ[ij]*(A_mat(ij.i,ij.j,𝐠,𝐱) + (1/(2*L))*C_mat(ij.i,ij.j,𝐠)) for ij in idx_set_λ) +
                sum(η[ij]*(⊙(-𝐠[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_η) +
                sum(ϕ[ij]*(⊙(-𝐠[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ϕ) +
                sum(κ[ij]*(⊙(𝐰[:,ij.j] - 𝐰[:,ij.i], 𝐰[:,ij.j] - 𝐰[:,ij.i])) for ij in idx_set_κ) +
                sum(ν[ij]*(⊙(𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ν) +
                sum(ρ[ij]*(⊙(𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ρ) +
                sum(ω[ij]*(⊙(𝐱[:,ij.i] - 𝐰[:,ij.i], 𝐱[:,ij.i] - 𝐰[:,ij.i])) for ij in idx_set_ω) +
                sum(σ[ij]*(⊙(𝐠[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_σ) +
                sum(χ[ij]*(⊙(-𝐠[:,ij.i], 𝐧[:,ij.j] - 𝐧[:,ij.i])) for ij in idx_set_χ) +
                sum(ζ[ij]*(⊙(𝐧[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_ζ) + 
                ξ*(⊙(𝐠[:,-1], 𝐠[:,-1]))
            .== Z
            )
        elseif functionType == :smoothSC
            @constraint(model,
                sum(λ[ij]*(L/(L-μ)*A_mat(ij.i,ij.j,𝐠,𝐱) + μ/(L-μ)*A_mat(ij.j,ij.i,𝐠,𝐱) + μ/(2*(1-μ/L))*B_mat(ij.i,ij.j,𝐱) + 1/(2*(L-μ))*C_mat(ij.i,ij.j,𝐠)) for ij in idx_set_λ) +
                sum(η[ij]*(⊙(-𝐠[:,ij.i], 𝐳[:,ij.j] - 𝐳[:,ij.i])) for ij in idx_set_η) +
                sum(ϕ[ij]*(⊙(-𝐠[:,ij.i], 𝐱[:,ij.j] - 𝐳[:,ij.i])) for ij in idx_set_ϕ) +
                sum(κ[ij]*(⊙(𝐰[:,ij.j] - 𝐰[:,ij.i], 𝐰[:,ij.j] - 𝐰[:,ij.i])) for ij in idx_set_κ) +
                sum(ν[ij]*(⊙(𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ν) +
                sum(ρ[ij]*(⊙(𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ρ) +
                sum(ω[ij]*(⊙(𝐱[:,ij.i] - 𝐰[:,ij.i], 𝐱[:,ij.i] - 𝐰[:,ij.i])) for ij in idx_set_ω) +
                sum(σ[ij]*(⊙(𝐠[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_σ) +
                sum(χ[ij]*(⊙(-𝐠[:,ij.i], 𝐧[:,ij.j] - 𝐧[:,ij.i])) for ij in idx_set_χ) +
                sum(ζ[ij]*(⊙(𝐧[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_ζ) + 
                ξ*(⊙(𝐠[:,-1], 𝐠[:,-1]))
            .== Z
            )
        else
            @warn "Invalid functionType"
        end
    else
        if functionType == :smooth
            @constraint(model,
                sum(λ[ij]*(A_mat(ij.i,ij.j,𝐠,𝐱) + (1/(2*L))*C_mat(ij.i,ij.j,𝐠)) for ij in idx_set_λ) +
                sum(η[ij]*(⊙(-𝐠[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_η) +
                sum(ϕ[ij]*(⊙(-𝐠[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ϕ) +
                sum(κ[ij]*(⊙(𝐰[:,ij.j] - 𝐰[:,ij.i], 𝐰[:,ij.j] - 𝐰[:,ij.i])) for ij in idx_set_κ) +
                sum(ν[ij]*(⊙(𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ν) +
                sum(ρ[ij]*(⊙(𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ρ) +
                sum(ω[ij]*(⊙(𝐱[:,ij.i] - 𝐰[:,ij.i], 𝐱[:,ij.i] - 𝐰[:,ij.i])) for ij in idx_set_ω) +
                sum(σ[ij]*(⊙(𝐠[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_σ) +
                sum(χ[ij]*(⊙(-𝐠[:,ij.i], 𝐧[:,ij.j] - 𝐧[:,ij.i])) for ij in idx_set_χ) +
                sum(ζ[ij]*(⊙(𝐧[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_ζ)
            .== Z
            )
        elseif functionType == :smoothSC
            @constraint(model,
                sum(λ[ij]*(L/(L-μ)*A_mat(ij.i,ij.j,𝐠,𝐱) + μ/(L-μ)*A_mat(ij.j,ij.i,𝐠,𝐱) + μ/(2*(1-μ/L))*B_mat(ij.i,ij.j,𝐱) + 1/(2*(L-μ))*C_mat(ij.i,ij.j,𝐠)) for ij in idx_set_λ) +
                sum(η[ij]*(⊙(-𝐠[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_η) +
                sum(ϕ[ij]*(⊙(-𝐠[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ϕ) +
                sum(κ[ij]*(⊙(𝐰[:,ij.j] - 𝐰[:,ij.i], 𝐰[:,ij.j] - 𝐰[:,ij.i])) for ij in idx_set_κ) +
                sum(ν[ij]*(⊙(𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐰[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ν) +
                sum(ρ[ij]*(⊙(𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i], 𝐳[:,ij.j] - 1/β*𝐧[:,ij.j] - 𝐳[:,ij.i] + 1/β*𝐧[:,ij.i])) for ij in idx_set_ρ) +
                sum(ω[ij]*(⊙(𝐱[:,ij.i] - 𝐰[:,ij.i], 𝐱[:,ij.i] - 𝐰[:,ij.i])) for ij in idx_set_ω) +
                sum(σ[ij]*(⊙(𝐠[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_σ) +
                sum(χ[ij]*(⊙(-𝐠[:,ij.i], 𝐧[:,ij.j] - 𝐧[:,ij.i])) for ij in idx_set_χ) +
                sum(ζ[ij]*(⊙(𝐧[:,ij.i], 𝐧[:,ij.i])) for ij in idx_set_ζ)
            .== Z
            )
        else
            @warn "Invalid functionType"
        end
    end
end


function localOptStepSolver_SmoothRelaxationFW(N, L, D, β,
    d_star_ws,                                  # warm start for dual solution value
    dualVars_ws,                                # warm start for dual variables
    stepVars_ws,                                # warm start for stepsize variables
    idx_sets_ws_eff,                            # warm start for index sets
    varBounds,                                  # upper bounds on dual variables
    functionType,                               # :smooth, :smoothSC
    primalObjective,                            # :minIterate, :finalIterate
    optLoc,                                     # :exterior, :inSet 
    stepMode;                                   # :standard, :matrix (:matrix corresponds to 'Generalized Frank-Wolfe' where x_{k+1} is a weighted sum of all z_k )
    #Options:
    print_model = :off,
    show_output = :off,                         # off, :on
    reduce_index_sets = :for_warm_start_only,
    reduce_index_set_for_Λ = :off,              # :off, :on
    find_global_lower_bound_via_cholesky_lazy_constraint = :off, # if this on, then we model Z = Λ*Λ^T via lazy constraint (the goal is to find a lower bound to StepDesign PEP)
    bound_impose = :off,                        # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation,
    quadratic_equality_modeling = :through_ϵ,   # other option is :exact
    cholesky_modeling = :formula,               # : formula implements the equivalent representation of Z = Λ*Λ^T via formulas, the other option is :definition, that directly model Z = Λ*Λ^T
    ϵ_tol_feas = 1e-6,                          # tolerance for feasibility
    ϵ_tol_Cholesky = 0.0005,                    # tolerance for determining which elements of Λ_ws is zero
    maxCutCount=1e3,                            # this is the number of cuts to be added if the lazy constraint callback is activated
    global_lower_bound_given = :off,            # whether is a global lower bound is given, providing this would make the branch-and-bound faster
    global_lower_bound = 0.0,                   # value of the global lower bound (if nothing is given then 0 is a valid lower bound)
    polish_solution = :on,                      # whether to polish the solution to get better precision, the other option is :off

)

    # Number of points
    # ----------------
    I = -1:N-1
    K = -1:N
    dim_Z = 2N+3

    # *************
    # declare model
    # -------------
    # *************

    model_optStep = Model(Ipopt.Optimizer)

    set_optimizer_attribute(model_optStep, "constr_viol_tol", 1e-7)
    set_optimizer_attribute(model_optStep, "dual_inf_tol", 1e-7)
    set_optimizer_attribute(model_optStep, "compl_inf_tol", 1e-7)

    set_optimizer_attribute(model_optStep, "tol", 1e-8)#
    set_optimizer_attribute(model_optStep, "max_iter", 100000)
    set_optimizer_attribute(model_optStep,"print_level",1)

    # ************************
    # define all the variables
    # ------------------------
    # ************************

    if functionType == :smoothSC
        @warn "smoothSC not currently supported"
        return -1
    end

    @info "---defining the variables"

    dualVars, stepVars, idx_sets, idx_sets_ws = defineStepDesignVariablesFW_SmoothRelaxation(model_optStep, N, primalObjective, optLoc, reduce_index_sets, idx_sets_ws_eff;
            find_global_lower_bound_via_cholesky_lazy_constraint
        )

    (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z, Λ) = dualVars
    (s, q, QQ, ψ) = stepVars
    (λ_ws, η_ws, ϕ_ws, κ_ws, ν_ws, ρ_ws, ω_ws, σ_ws, χ_ws, ζ_ws, τ_ws, ξ_ws, Z_ws, Λ_ws) = dualVars_ws
    (s_ws, ψ_ws) = stepVars_ws
    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ, idx_set_s, idx_set_q, idx_set_QQ, idx_set_ψ) = idx_sets
    (idx_set_λ_ws, idx_set_η_ws, idx_set_ϕ_ws, idx_set_σ_ws, idx_set_χ_ws) = idx_sets_ws
    (M_λ, M_η, M_ϕ, M_κ, M_ν, M_ρ, M_ω, M_σ, M_χ, M_ζ, M_Z, M_Λ) = varBounds

    # Apply constraints specific to linking certain variables or fixing certain values
    applyVariableConstraints_SmoothRelaxation(model_optStep, N, stepVars, idx_sets, stepMode)


    # warm-start values for all the variables
    # ----------------------------------------------------

    @info "---warm-start values for all the variables"

    warmStartStepDesignFW_SmoothRelaxation(reduce_index_sets, N, primalObjective, idx_sets, idx_sets_ws, dualVars, stepVars, dualVars_ws, stepVars_ws;
        find_global_lower_bound_via_cholesky_lazy_constraint = find_global_lower_bound_via_cholesky_lazy_constraint)

    # ************
    # ---add objective
    # -------------
    # *************

    @info "---adding objective"

    setStepDesignObjectiveFW_SmoothRelaxation(model_optStep, d_star_ws, D, β, dualVars, idx_sets;
        global_lower_bound_given = :on,
        global_lower_bound = global_lower_bound)

    # ******************************
    # ---add the data generator function
    # *******************************

    @info "---adding the data generator function to create 𝐱, 𝐠, 𝐟"

    𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetSmoothRelaxationStepDesign(N, q)

    # *******************
    # add the constraints
    # *******************


    # add the linear constraint
    # -------------------------

    @info "---adding linear constraint"

    addStepDesignLinearConstraintFW_SmoothRelaxation(model_optStep, primalObjective, N, λ, idx_set_λ, τ, 𝐟)

    # add the LMI constraint
    # ----------------------

    @info "---adding LMI constraint"


    addStepDesignLMIConstraintFW_SmoothRelaxation(model_optStep, functionType, optLoc, N, L, β, 𝐠, 𝐱, 𝐳, 𝐧, 𝐰, dualVars, stepVars, idx_sets)


    # implementation through ϵ_tol_feas

    # add valid constraints for Z ⪰ 0
    # -------------------------------

    @info "---adding valid constraints for Z"

    # diagonal components of Z are non-negative
    for i in 1:dim_Z
        @constraint(model_optStep, Z[i,i] >= 0)
    end

    # the off-diagonal components satisfy:
    # (∀i,j ∈ dim_Z: i != j) -(0.5*(Z[i,i] + Z[j,j])) <= Z[i,j] <=  (0.5*(Z[i,i] + Z[j,j]))

    for i in 1:dim_Z
        for j in 1:dim_Z
            if i != j
                @constraint(model_optStep, Z[i,j] <= (0.5*(Z[i,i] + Z[j,j])) )
                @constraint(model_optStep, -(0.5*(Z[i,i] + Z[j,j])) <= Z[i,j] )
            end
        end
    end

    # add cholesky related constraints
    # --------------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off

        @info "---adding cholesky matrix related constraints"

        # Two constraints to define the matrix Λ to be a lower triangular matrix
        # -------------------------------------------------

        # upper off-diagonal terms of Λ are zero

        for i in 1:dim_Z
            for j in 1:dim_Z
                if i < j
                    # @constraint(model_optStep, Λ[i,j] .== 0)
                    fix(Λ[i,j], 0; force = true)
                end
            end
        end

        # diagonal components of Λ are non-negative

        for i in 1:dim_Z
            @constraint(model_optStep, Λ[i,i] >= 0)
        end

    end

    # time to implement Z = L*L^T constraint
    # --------------------------------------

    if cholesky_modeling == :definition && find_global_lower_bound_via_cholesky_lazy_constraint == :off

        if quadratic_equality_modeling == :exact

            # direct modeling through definition and vectorization
            # ---------------------------------------------------
            @constraint(model_optStep, vectorize(Z - (Λ * Λ'), SymmetricMatrixShape(dim_Z)) .== 0)

        elseif quadratic_equality_modeling == :through_ϵ

            # definition modeling through vectorization and ϵ_tol_feas

            # part 1: models Z-Λ*Λ <= ϵ_tol_feas*ones(dim_Z,dim_Z)
            @constraint(model_optStep, vectorize(Z - (Λ * Λ') - ϵ_tol_feas*ones(dim_Z,dim_Z), SymmetricMatrixShape(dim_Z)) .<= 0)

            # part 2: models Z-Λ*Λ >= -ϵ_tol_feas*ones(dim_Z,dim_Z)

            @constraint(model_optStep, vectorize(Z - (Λ * Λ') + ϵ_tol_feas*ones(dim_Z,dim_Z), SymmetricMatrixShape(dim_Z)) .>= 0)

        else

            @error "something is not right in Cholesky modeling"

            return

        end


    elseif cholesky_modeling == :formula && find_global_lower_bound_via_cholesky_lazy_constraint == :off

        # Cholesky constraint 1
        # (∀ j ∈ dim_Z) Λ[j,j]^2 + ∑_{k∈[1:j-1]} Λ[j,k]^2 == Z[j,j]

        for j in 1:dim_Z
            if j == 1
                @constraint(model_optStep, Λ[j,j]^2 == Z[j,j])
            elseif j > 1
                @constraint(model_optStep, Λ[j,j]^2+sum(Λ[j,k]^2 for k in 1:j-1) == Z[j,j])
            end
        end

        # Cholesky constraint 2
        # (∀ i,j ∈ dim_Z: i > j) Λ[i,j] Λ[j,j] + ∑_{k∈[1:j-1]} Λ[i,k] Λ[j,k] == Z[i,j]

        for i in 1:dim_Z
            for j in 1:dim_Z
                if i>j
                    if j == 1
                        @constraint(model_optStep, Λ[i,j]*Λ[j,j]  == Z[i,j])
                    elseif j > 1
                        @constraint(model_optStep, Λ[i,j]*Λ[j,j] + sum(Λ[i,k]*Λ[j,k] for k in 1:j-1) == Z[i,j])
                    end
                end
            end
        end

    elseif find_global_lower_bound_via_cholesky_lazy_constraint == :on

        # set_optimizer_attribute(model_optStep, "FuncPieces", -2) # FuncPieces = -2: Bounds the relative error of the approximation; the error bound is provided in the FuncPieceError attribute. See https://www.gurobi.com/documentation/9.1/refman/funcpieces.html#attr:FuncPieces

        # set_optimizer_attribute(model_optStep, "FuncPieceError", 0.1) # relative error

        set_optimizer_attribute(model_optStep, "MIPFocus", 1) # focus on finding good quality feasible solution

        # add initial cuts
        num_cutting_planes_init = 2*dim_Z^2
        cutting_plane_array = randn(dim_Z,num_cutting_planes_init)
        num_cuts_array_rows, num_cuts = size(cutting_plane_array)
        for i in 1:num_cuts
            d_cut = cutting_plane_array[:,i]
            d_cut = d_cut/norm(d_cut,2) # normalize the cutting plane vector
            @constraint(model_optStep, tr(Z*(d_cut*d_cut')) >= 0)
        end

        cutCount=0
        # maxCutCount=1e3

        # add the lazy callback function
        # ------------------------------
        function add_lazy_callback(cb_data)
            if cutCount<=maxCutCount
                Z0 = zeros(dim_Z,dim_Z)
                for i=1:dim_Z
                    for j=1:dim_Z
                        Z0[i,j]=callback_value(cb_data, Z[i,j])
                    end
                end
                if eigvals(Z0)[1]<=-0.01
                    u_t = eigvecs(Z0)[:,1]
                    u_t = u_t/norm(u_t,2)
                    con3 = @build_constraint(tr(Z*u_t*u_t') >=0.0)
                    MOI.submit(model_optStep, MOI.LazyConstraint(cb_data), con3)
                    # noPSDCuts+=1
                end
                cutCount+=1
            end
        end

        # submit the lazy constraint
        # --------------------------
        MOI.set(model_optStep, MOI.LazyConstraintCallback(), add_lazy_callback)


    end

    # impose bound on the variables if bound_impose == :on

    if bound_impose == :on
        @info "---finding bound on the variables"

        # store the values

        λ_lb, η_lb, ϕ_lb, κ_lb, ν_lb, ρ_lb, ω_lb, σ_lb, χ_lb, ζ_lb = 0,0,0,0,0,0,0,0,0,0
        λ_ub, η_ub, ϕ_ub, κ_ub, ν_ub, ρ_ub, ω_ub, σ_ub, χ_ub, ζ_ub = M_λ, M_η, M_ϕ, M_κ, M_ν, M_ρ, M_ω, M_σ, M_χ, M_ζ
        Z_lb = -M_Z
        Z_ub = M_Z
        Λ_lb = -M_Λ
        Λ_ub = M_Λ

        # set bound for vars
        # ---------------
        # set_lower_bound.([], []_lb): done in definition
        set_upper_bound.(λ, λ_ub)
        set_upper_bound.(η, η_ub)
        set_upper_bound.(ϕ, ϕ_ub)
        set_upper_bound.(κ, κ_ub)
        set_upper_bound.(ν, ν_ub)
        set_upper_bound.(ρ, ρ_ub)
        set_upper_bound.(ω, ω_ub)
        set_upper_bound.(σ, σ_ub)
        set_upper_bound.(χ, χ_ub)
        set_upper_bound.(ζ, ζ_ub)

        # set bound for Z
        # ---------------
        for i in 1:dim_Z
            for j in 1:dim_Z
                set_lower_bound(Z[i,j], Z_lb)
                set_upper_bound(Z[i,j], Z_ub)
            end
        end

        # set bound for Λ
        # ------------------------

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off
            # need only upper bound for the diagonal components, as the lower bound is zero from the model
            for i in 1:N+2
                set_upper_bound(Λ[i,i], Λ_ub)
            end
            # need to bound only components, Λ[i,j] with i > j, as for i < j, we have zero, due to the lower triangular structure
            for i in 1:N+2
                for j in 1:N+2
                    if i > j
                        set_lower_bound(Λ[i,j], Λ_lb)
                        set_upper_bound(Λ[i,j], Λ_ub)
                    end
                end
            end
        end

        for idx in idx_set_ψ
            set_upper_bound(ψ[idx], 1)
        end

        if primalObjective == :minIterate
            for i in 0:N
                set_upper_bound(τ[i+1], 1)
            end
        end
        

    end

    # impose the effective index set of Λ if reduce_index_set_for_Λ  == :on and we are not computing a global lower bound
    # ------------------------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off && reduce_index_set_for_Λ == :on
        zis_Lc = zero_index_set_finder_Λ(Λ_ws; ϵ_tol = ϵ_tol_Cholesky)
        for k in 1:length(zis_Lc)
            fix(Λ[CartesianIndex(zis_Lc[k])], 0; force = true)
        end
    end


    # time to optimize
    # ----------------

    @info "---model building done, starting the optimization process"

    if show_output == :off
        set_silent(model_optStep)
    end

    optimize!(model_optStep)

    @info "model_optStep has termination status = " termination_status(model_optStep)

    fail = 0
    if termination_status(model_optStep) in [MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED]

        # store the solutions and return
        # ------------------------------

        @info "---optimal solution found done, store the solution"

        # store opt vals

        λ_opt = value.(λ)
        η_opt = value.(η)
        ϕ_opt = value.(ϕ)
        κ_opt = value.(κ)
        ν_opt = value.(ν)
        ρ_opt = value.(ρ)
        ω_opt = value.(ω)
        σ_opt = value.(σ)
        χ_opt = value.(χ)
        ζ_opt = value.(ζ)

        # store h_opt

        s_opt = value.(s)
        q_opt = value.(q)
        QQ_opt = value.(QQ)
        ψ_opt = value.(ψ)

        if primalObjective == :minIterate
            τ_opt = value.(τ)
        else
            τ_opt = 0
        end

        if optLoc == :inSet
            ξ_opt = value.(ξ)
        else
            ξ_opt = 0
        end

        # store Z_opt

        Z_opt = value.(Z)

        # store Λ

        if find_global_lower_bound_via_cholesky_lazy_constraint == :off

            Λ_opt = value.(Λ)

            if norm(Z_opt - Λ_opt*Λ_opt', Inf) > 10^-4
                @warn "||Z - Λ*Λ^T|| = $(norm(Z_opt -  Λ_opt*Λ_opt', Inf))"
            end

        elseif find_global_lower_bound_via_cholesky_lazy_constraint == :on

            Λ_opt = compute_pivoted_cholesky_L_mat(Z_opt; suppressWarning = true)

            # in this case doing the cholesky check does not make sense, because we are not aiming to find a psd Z_opt

            # if norm(Z_opt - Λ_opt*Λ_opt', Inf) > 10^-4
            #     @info "checking the norm bound"
            #     @warn "||Z - L*L^T|| = $(norm(Z_opt - Λ_opt*Λ_opt', Inf))"
            # end

        end

        obj_val = objective_value(model_optStep)

    else

        @warn "---could not find an optimal solution, returning the warm-start point"

        fail = 1
        obj_val, λ_opt, η_opt, ϕ_opt, κ_opt, ν_opt, ρ_opt, ω_opt, σ_opt, χ_opt, ζ_opt, s_opt, q_opt, QQ_opt, ψ_opt, τ_opt, ξ_opt, Z_opt, Λ_opt, idx_set_λ_opt_eff, idx_set_η_opt_eff, idx_set_ϕ_opt_eff, idx_set_σ_opt_eff, idx_set_χ_opt_eff = -1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1

    end

    # find the effective index set of the found λ, ψ

    if fail == 1
        idx_set_λ_opt_eff = idx_set_λ 
        idx_set_η_opt_eff = idx_set_η
        idx_set_ϕ_opt_eff = idx_set_ϕ
        idx_set_σ_opt_eff = idx_set_σ
        idx_set_χ_opt_eff = idx_set_χ
    else
        idx_set_λ_opt_eff = effective_index_set_finder(λ_opt ; ϵ_tol = 1e-6)
        idx_set_η_opt_eff = effective_index_set_finder(η_opt ; ϵ_tol = 1e-6)
        idx_set_ϕ_opt_eff = effective_index_set_finder(ϕ_opt ; ϵ_tol = 1e-6)
        idx_set_σ_opt_eff = effective_index_set_finder(σ_opt ; ϵ_tol = 1e-6)
        idx_set_χ_opt_eff = effective_index_set_finder(χ_opt ; ϵ_tol = 1e-6)
    end



    @info "---warm-start objective value = $d_star_ws, and objective value of found solution = $obj_val"


    if print_model == :on
        print(model_optStep)
    end

    # time to return all the stored values

    dualVars_opt = (λ_opt, η_opt, ϕ_opt, κ_opt, ν_opt, ρ_opt, ω_opt, σ_opt, χ_opt, ζ_opt, τ_opt, ξ_opt, Z_opt, Λ_opt)
    stepVars_opt = (s_opt, q_opt, QQ_opt, ψ_opt)
    idx_sets_opt_eff = (idx_set_λ_opt_eff, idx_set_η_opt_eff, idx_set_ϕ_opt_eff, idx_set_σ_opt_eff, idx_set_χ_opt_eff)

    return obj_val, dualVars_opt, stepVars_opt, idx_sets_opt_eff, model_optStep


end

function defineStepDesignVariablesFW_SmoothRelaxation(model, N, primalObjective, optLoc, reduce_index_sets, idx_sets_ws_eff;
    find_global_lower_bound_via_cholesky_lazy_constraint)

    (idx_set_λ_ws_eff, idx_set_η_ws_eff, idx_set_ϕ_ws_eff, idx_set_σ_ws_eff, idx_set_χ_ws_eff) = idx_sets_ws_eff

    I = -1:N-1
    K = -1:N
    dim_Z = 4N+6

    if reduce_index_sets == :off
        # define vars over the full index set

        #warm-start index arrays not needed, so set to null so we have a value to return
        idx_set_λ_ws, idx_set_η_ws, idx_set_ϕ_ws = [], [], []

        idx_set_λ = index_set_constructor(K)        # Constraint: f Smooth convex
        idx_set_η = index_set_constructor(I)        # Constraint: Smooth convex
        idx_set_ϕ = index_set_constructor(I,K)      # Constraint: w ∈ Set
        idx_set_κ = index_set_constructor_symm(K) # Constraint: w-w diameter
        idx_set_ν = index_set_constructor(I,K)   # Constraint: w-z_shift diameter
        idx_set_ρ = index_set_constructor_symm(I)   # Constraint: z_shift-z_shift diameter
        idx_set_ω = index_set_constructor_single(K)   # Constraint: x-w link
        idx_set_σ = index_set_constructor_single(I)   # Constraint g-n alignment
        idx_set_χ = index_set_constructor(I)            # Constraint: g-n relaxation
        idx_set_ζ = index_set_constructor_single(I)     #Constraint: n normalization

    elseif reduce_index_sets == :on
        # define vars over a reduced index set, idx_set_<>_ws_effective, which is the effective index set of <>_ws

        #warm-start index arrays not needed, so set to null so we have a value to return
        idx_set_λ_ws, idx_set_η_ws, idx_set_ϕ_ws = [], [], []

        idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_σ, idx_set_χ = idx_set_λ_ws_eff, idx_set_η_ws_eff, idx_set_ϕ_ws_eff, idx_set_σ_ws_eff, idx_set_χ_ws_eff
        idx_set_κ = index_set_constructor_symm(K) # Constraint: w-w diameter
        idx_set_ν = index_set_constructor(I,K)   # Constraint: w-z_shift diameter
        idx_set_ρ = index_set_constructor_symm(I)   # Constraint: z_shift-z_shift diameter
        idx_set_ω = index_set_constructor_single(K)   # Constraint: x-w link
        idx_set_ζ = index_set_constructor_single(I)     #Constraint: n normalization

    elseif reduce_index_sets == :for_warm_start_only
        # this :for_warm_start_only option is same as the :off option, however in this case we will define λ (and ψ) over the full index set, but warm,-start from a λ_ws that has reduced index set
        idx_set_λ = index_set_constructor(K)        # Constraint: f Smooth convex
        idx_set_η = index_set_constructor(I)        # Constraint: Smooth convex
        idx_set_ϕ = index_set_constructor(I,K)      # Constraint: w ∈ Set
        idx_set_κ = index_set_constructor_symm(K) # Constraint: w-w diameter
        idx_set_ν = index_set_constructor(I,K)   # Constraint: w-z_shift diameter
        idx_set_ρ = index_set_constructor_symm(I)   # Constraint: z_shift-z_shift diameter
        idx_set_ω = index_set_constructor_single(K)   # Constraint: x-w link
        idx_set_σ = index_set_constructor_single(I)   # Constraint g-n alignment
        idx_set_χ = index_set_constructor(I)    # Constraint: g-n relaxation
        idx_set_ζ = index_set_constructor_single(I)     #Constraint: n normalization

        idx_set_λ_ws, idx_set_η_ws, idx_set_ϕ_ws, idx_set_σ_ws, idx_set_χ_ws = idx_set_λ_ws_eff, idx_set_η_ws_eff, idx_set_ϕ_ws_eff, idx_set_σ_ws_eff, idx_set_χ_ws_eff

    end

    @variable(model, λ[idx_set_λ] >= 0)
    @variable(model, η[idx_set_η] >= 0)
    @variable(model, ϕ[idx_set_ϕ] >= 0)
    @variable(model, κ[idx_set_κ] >= 0)
    @variable(model, ν[idx_set_ν] >= 0)
    @variable(model, ρ[idx_set_ρ] >= 0)
    @variable(model, ω[idx_set_ω] >= 0)
    @variable(model, σ[idx_set_σ] >= 0)
    @variable(model, χ[idx_set_χ] >= 0)
    @variable(model, ζ[idx_set_ζ] >= 0)


    # define Z
    # --------

    @variable(model, Z[1:dim_Z, 1:dim_Z], Symmetric)

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off

        # define the cholesky matrix of Z: Λ
        # -------------------------------------------
        @variable(model, Λ[1:dim_Z, 1:dim_Z])

    else
        Λ = 0
    end

    # define τ for :minIterate
    if primalObjective == :minIterate
        @variable(model, τ[1:N+1] >= 0)
    elseif primalObjective == :finalIterate
        τ = 0
    else
        @warn "Invalid primalObjective"
    end

    if optLoc == :inSet
        @variable(model, ξ >= 0)
    else
        ξ = 0
    end

    # define the stepsequence matrix
    # ----------------------------
    idx_set_s = index_set_constructor_lowertri(0:N-1)
    @variable(model, s[idx_set_s] >= 0)

    idx_set_q = index_set_constructor_lowertri(0:N-1)
    @variable(model, q[idx_set_q] >= 0)

    idx_set_ψ = index_set_constructor_single(0:N-1)
    @variable(model, ψ[idx_set_ψ] >= 0)

    idx_set_QQ = index_set_constructor_QQ(N)
    @variable(model, QQ[idx_set_QQ] >= 0)

    dualVars = (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z, Λ)
    stepVars = (s, q, QQ, ψ)
    idx_sets = (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ, idx_set_s, idx_set_q, idx_set_QQ, idx_set_ψ)
    idx_sets_ws = (idx_set_λ_ws, idx_set_η_ws, idx_set_ϕ_ws, idx_set_σ_ws, idx_set_χ_ws)

    return dualVars, stepVars, idx_sets, idx_sets_ws

end

# Apply constraints specific to linking certain variables or fixing certain values
function applyVariableConstraints_SmoothRelaxation(model, N, stepVars, idx_sets, stepMode)

    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ, idx_set_s, idx_set_q, idx_set_QQ, idx_set_ψ) = idx_sets
    (s, q, QQ, ψ) = stepVars
    
    # Enforces x1 = z0, which we want. It will probably help the code run faster by eliminating really bad other cases
    @constraint(model, s[ij_idx(0,0)] == 1.0)
    @constraint(model, ψ[i_idx(0)] == 1.0)
    

    # Link q variable to s and ψ
    for ij_q in idx_set_q
        @constraint(model, ψ[i_idx(ij_q.i)]*s[ij_q] == q[ij_q])
    end

    # Link QQ variables to q
    for idx in idx_set_QQ
        @constraint(model, QQ[ijk_idx(idx.i, idx.j, idx.k)] == q[ij_idx(idx.i, idx.j)]*q[ij_idx(idx.i,idx.k)])
    end

    # Sum q = 1 - This enforces ψ as reciprocal
    for i=1:N-1     #Start at i=1, since o=0 is implicit from above
        @constraint(model, sum(q[ij_idx(i,j)] for j in 0:i) == 1)
    end

    # Columns of s are equal: s_ij = s_jj for i>j
    if stepMode == :standard
        for ij_s in idx_set_s
            if ij_s.i <= ij_s.j
                continue
            end
            @constraint(model, s[ij_s] == s[ij_idx(ij_s.j, ij_s.j)])
        end
    end

end



function warmStartStepDesignFW_SmoothRelaxation(reduce_index_sets, N, primalObjective,
    # Index sets
    # ----------
    idx_sets, idx_sets_ws,
    # variables
    #----------
    dualVars, stepVars,
    # Warm-start values
    #------------------
    dualVars_ws, stepVars_ws;
    # Settings
    # --------------
    find_global_lower_bound_via_cholesky_lazy_constraint = :off
    )

    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ, idx_set_s, idx_set_q, idx_set_QQ, idx_set_ψ) = idx_sets
    (idx_set_λ_ws, idx_set_η_ws, idx_set_ϕ_ws, idx_set_σ_ws, idx_set_χ_ws) = idx_sets_ws
    (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z, Λ) = dualVars
    (s, q, QQ, ψ) = stepVars
    (λ_ws, η_ws, ϕ_ws, κ_ws, ν_ws, ρ_ws, ω_ws, σ_ws, χ_ws, ζ_ws, τ_ws, ξ_ws, Z_ws, Λ_ws) = dualVars_ws
    (s_ws, ψ_ws) = stepVars_ws


    dim_Z = size(Z,1)

    # warm start for λ
    # ----------------
    if reduce_index_sets == :for_warm_start_only
        for ij_λ in idx_set_λ_ws
            set_start_value(λ[ij_λ], λ_ws[ij_λ])
        end
        for ij_λ in setdiff(idx_set_λ, idx_set_λ_ws)
            set_start_value(λ[ij_λ], 0.0)
        end
    else
        for ij_λ in idx_set_λ
            set_start_value(λ[ij_λ], λ_ws[ij_λ])
        end
    end

    # warm start for η
    # ----------------
    if reduce_index_sets == :for_warm_start_only
        for ij_η in idx_set_η_ws
            set_start_value(η[ij_η], η_ws[ij_η])
        end
        for ij_η in setdiff(idx_set_η, idx_set_η_ws)
            set_start_value(η[ij_η], 0.0)
        end
    else
        for ij_η in idx_set_η
            set_start_value(η[ij_η], η_ws[ij_η])
        end
    end

    # warm start for ϕ
    # ----------------
    if reduce_index_sets == :for_warm_start_only
        for ij_ϕ in idx_set_ϕ_ws
            set_start_value(ϕ[ij_ϕ], ϕ_ws[ij_ϕ])
        end
        for ij_ϕ in setdiff(idx_set_ϕ, idx_set_ϕ_ws)
            set_start_value(ϕ[ij_ϕ], 0.0)
        end
    else
        for ij_ϕ in idx_set_ϕ
            set_start_value(ϕ[ij_ϕ], ϕ_ws[ij_ϕ])
        end
    end

    # warm start for σ
    # ----------------
    if reduce_index_sets == :for_warm_start_only
        for ij_σ in idx_set_σ_ws
            set_start_value(σ[ij_σ], σ_ws[ij_σ])
        end
        for ij_σ in setdiff(idx_set_σ, idx_set_σ_ws)
            set_start_value(σ[ij_σ], 0.0)
        end
    else
        for ij_σ in idx_set_σ
            set_start_value(σ[ij_σ], σ_ws[ij_σ])
        end
    end
    

    # warm start for χ
    # ----------------
    if reduce_index_sets == :for_warm_start_only
        for ij_χ in idx_set_χ_ws
            set_start_value(χ[ij_χ], χ_ws[ij_χ])
        end
        for ij_χ in setdiff(idx_set_χ, idx_set_χ_ws)
            set_start_value(χ[ij_χ], 0.0)
        end
    else
        for ij_χ in idx_set_χ
            set_start_value(χ[ij_χ], χ_ws[ij_χ])
        end
    end


    # warm start for κ
    # ----------------
    for ij_κ in idx_set_κ
        set_start_value(κ[ij_κ], κ_ws[ij_κ])
    end

    # warm start for ν
    # ----------------
    for ij_ν in idx_set_ν
        set_start_value(ν[ij_ν], ν_ws[ij_ν])
    end

    # warm start for ρ
    # ----------------
    for ij_ρ in idx_set_ρ
        set_start_value(ρ[ij_ρ], ρ_ws[ij_ρ])
    end

    # warm start for ω
    # ----------------
    for idx in idx_set_ω
        set_start_value(ω[idx], ω_ws[idx])
    end

    # warm start for ζ
    # ----------------
    for idx in idx_set_ζ
        set_start_value(ζ[idx], ζ_ws[idx])
    end

    # warm start for Z
    # ----------------

    for i in 1:dim_Z
        for j in 1:dim_Z
            set_start_value(Z[i,j], Z_ws[i,j])
        end
    end

    # warm start for Λ
    # ------------------------

    if find_global_lower_bound_via_cholesky_lazy_constraint == :off
        for i in 1:dim_Z
            for j in 1:dim_Z
                set_start_value(Λ[i,j], Λ_ws[i,j])
            end
        end
    end

    # warm start for s,q,ψ
    # ----------------
    for ij_s in idx_set_s
        try 
            set_start_value(s[ij_s], s_ws[ij_s.i, ij_s.j])
        catch e
            set_start_value(s[ij_s], s_ws[ij_s])
        end
    end

    for ij_ψ in idx_set_ψ
        try 
            set_start_value(ψ[ij_ψ], ψ_ws[ij_ψ.i])
        catch e
            set_start_value(ψ[ij_ψ], ψ_ws[ij_ψ])
        end
    end

    for ij_q in idx_set_q
        try 
            set_start_value(q[ij_q], ψ_ws[ij_q.i]*s_ws[ij_q.i, ij_q.j])
        catch e
            set_start_value(q[ij_q], ψ_ws[i_idx(ij_q.i)]*s_ws[ij_q])
        end
    end

    for idx in idx_set_QQ
        try
            set_start_value(QQ[idx], (ψ_ws[idx.i]*s_ws[idx.i, idx.j])*(ψ_ws[idx.i]*s_ws[idx.i, idx.k]))
        catch e
            set_start_value(QQ[idx], (ψ_ws[i_idx(idx.i)]*s_ws[ij_idx(idx.i,idx.j)])*(ψ_ws[i_idx(idx.i)]*s_ws[ij_idx(idx.i, idx.k)]) )
        end
    end

    if primalObjective == :minIterate
        for i in 0:N
            set_start_value(τ[i+1], τ_ws[i+1])
        end
    end

    if optLoc == :inSet
        set_start_value(ξ, ξ_ws)
    end

end



function setStepDesignObjectiveFW_SmoothRelaxation(model, d_star_ws, D, β, dualVars, idx_sets; global_lower_bound_given = :off, global_lower_bound = 0.0)

    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ, idx_set_s, idx_set_q, idx_set_QQ, idx_set_ψ) = idx_sets
    (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z, Λ) = dualVars
    
    # Set objective
    @objective(model, Min, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (D-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)))
    # Add upper bound for objective
    @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (D-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) <= 1.001*d_star_ws) # this 1.001 factor gives some slack
    # Add a lower bound for objective (if given)
    if global_lower_bound_given == :on
        @constraint(model, sum(ζ[ij] for ij in idx_set_ζ) + 1/β^2*sum(ω[ij] for ij in idx_set_ω) + (D-2/β)^2*( sum(κ[ij] for ij in idx_set_κ)  +  sum(ν[ij] for ij in idx_set_ν)  +  sum(ρ[ij] for ij in idx_set_ρ)) >= global_lower_bound)
    end

end



function data_generator_SetSmoothRelaxationStepDesign(N, q)

    dim_G = 4N+6
    dim_F = N+1

    
    # define all the bold vectors
    # --------------------------


    # define 𝐱_0 and 𝐱_star


    𝐱_0 = e_i(dim_G, 1)

    𝐱_star = zeros(dim_G, 1)

    # define 𝐠_⋆ 𝐠_0, 𝐠_1, …, 𝐠_N

    # first we define the 𝐠 vectors,
    # index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

    # 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]
    𝐠 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in -1:N
        𝐠[:,i] = e_i(dim_G, i+3)
    end

    # 𝐳 = [𝐳_0 𝐳_1 𝐳_2 ... 𝐳_N-1]
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in 0:N-1
        𝐳[:,i] = e_i(dim_G, i+(N+4)) #Shift by N+4 to account for the 𝐠 vectors
    end

    # 𝐧 = [𝐧_⋆ 𝐧_0 𝐧_1 𝐧_2 ... 𝐧_N-1]
    𝐧 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in -1:N-1
        𝐧[:,i] = e_i(dim_G, i+(2N+5)) #Shift by 2N+4(+1) to account for the 𝐠 and 𝐳 vectors
    end

    # 𝐰 = [𝐰_⋆ 𝐰_0 𝐰_1 𝐰_2 ... 𝐰_N]
    𝐰 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in -1:N
        𝐰[:,i] = e_i(dim_G, i+(3N+6)) #Shift by 3N+4 (+1) to account for the 𝐠 and 𝐳 vectors
    end


    # time to define the 𝐟 vectors

    # 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_F, N+2), 1:dim_F, -1:N)

    for i in 0:N
        𝐟[:,i] = e_i(dim_F, i+1)
    end

    # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}]
    𝐱 = [𝐱_star 𝐱_0]

    for k=1:N
        𝐱_k = sum(q[ij_idx(k-1,j)]*𝐳[:,j] for j in 0:k-1)
        𝐱 = [𝐱   𝐱_k]
    end

    #NOTE: Offset has to happen last. Appending will mess it up
    # make 𝐱 an offset array to make our life comfortable
    𝐱 = OffsetArray(𝐱, 1:dim_G, -1:N)

    return 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟

end



function addStepDesignLinearConstraintFW_SmoothRelaxation(model, primalObjective, N, λ, idx_set_λ, τ, 𝐟)
    addDualLinearConstraintFW_SmoothRelaxation(model, N, primalObjective, λ, idx_set_λ, τ, 𝐟)  
end



function addStepDesignLMIConstraintFW_SmoothRelaxation(model, functionType, optLoc, N, L, β, 𝐠, 𝐱, 𝐳, 𝐧, 𝐰, dualVars, stepVars, idx_sets)
    
    (λ, η, ϕ, κ, ν, ρ, ω, σ, χ, ζ, τ, ξ, Z, Λ) = dualVars
    (s, q, QQ, ψ) = stepVars
    (idx_set_λ, idx_set_η, idx_set_ϕ, idx_set_κ, idx_set_ν, idx_set_ρ, idx_set_ω, idx_set_σ, idx_set_χ, idx_set_ζ, idx_set_s, idx_set_q, idx_set_QQ, idx_set_ψ) = idx_sets

    dim_Z = 4N+6

    𝐱_star = zeros(dim_Z,1)
    𝐱_0 = e_i(dim_Z,1)
    BMat = 𝐱_0*𝐱_0'


    if optLoc == :inSet
        @constraint(model,
        vectorize(
            sum(λ[idx]*(A_mat(idx.i,idx.j,𝐠,𝐱) + 1/(2L)*C_mat(idx.i,idx.j,𝐠)) for idx in idx_set_λ) +
            sum(η[idx]*(⊙(-𝐠[:,idx.i], 𝐳[:,idx.j] - 1/β*𝐧[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_η) +
            sum(ϕ[idx]*(⊙(-𝐠[:,idx.i], 𝐰[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_ϕ) +
            sum(κ[idx]*(⊙(𝐰[:,idx.j] - 𝐰[:,idx.i], 𝐰[:,idx.j] - 𝐰[:,idx.i])) for idx in idx_set_κ) +
            sum(ν[idx]*(⊙(𝐰[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i], 𝐰[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_ν) +
            sum(ρ[idx]*(⊙(𝐳[:,idx.j] - 1/β*𝐧[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i], 𝐳[:,idx.j] - 1/β*𝐧[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_ρ) +
            
            #Handle -1 and 0 normally
            ω[i_idx(-1)]*(⊙(𝐱_star - 𝐰[:,-1],  𝐱_star - 𝐰[:,-1])) + 
            ω[i_idx(0)]*(⊙(𝐱_0 - 𝐰[:,0],  𝐱_0 - 𝐰[:,0])) +
            # Special handling for 1:N
            sum(ω[i_idx(i)]*specialMatQQ(QQ, q, i, 𝐳, 𝐰) for i in 1:N) +

            sum(σ[idx]*(⊙(𝐠[:,idx.i], 𝐧[:,idx.i])) for idx in idx_set_σ) +
            sum(χ[idx]*(⊙(-𝐠[:,idx.i], 𝐧[:,idx.j] - 𝐧[:,idx.i])) for idx in idx_set_χ) +
            sum(ζ[idx]*(⊙(𝐧[:,idx.i], 𝐧[:,idx.i])) for idx in idx_set_ζ) + 
            ξ*(⊙(𝐠[:,-1], 𝐠[:,-1]))     # If optLoc=:inSet, include ξ variable
            - Z,
            SymmetricMatrixShape(dim_Z)
            ) .== 0
        )

    else
        @constraint(model,
        vectorize(
            sum(λ[idx]*(A_mat(idx.i,idx.j,𝐠,𝐱) + 1/(2L)*C_mat(idx.i,idx.j,𝐠)) for idx in idx_set_λ) +
            sum(η[idx]*(⊙(-𝐠[:,idx.i], 𝐳[:,idx.j] - 1/β*𝐧[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_η) +
            sum(ϕ[idx]*(⊙(-𝐠[:,idx.i], 𝐰[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_ϕ) +
            sum(κ[idx]*(⊙(𝐰[:,idx.j] - 𝐰[:,idx.i], 𝐰[:,idx.j] - 𝐰[:,idx.i])) for idx in idx_set_κ) +
            sum(ν[idx]*(⊙(𝐰[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i], 𝐰[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_ν) +
            sum(ρ[idx]*(⊙(𝐳[:,idx.j] - 1/β*𝐧[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i], 𝐳[:,idx.j] - 1/β*𝐧[:,idx.j] - 𝐳[:,idx.i] + 1/β*𝐧[:,idx.i])) for idx in idx_set_ρ) +
            
            #Handle -1 and 0 normally
            ω[i_idx(-1)]*(⊙(𝐱_star - 𝐰[:,-1],  𝐱_star - 𝐰[:,-1])) + 
            ω[i_idx(0)]*(⊙(𝐱_0 - 𝐰[:,0],  𝐱_0 - 𝐰[:,0])) +
            # Special handling for 1:N
            sum(ω[i_idx(i)]*specialMatQQ(QQ, q, i, 𝐳, 𝐰) for i in 1:N) +

            sum(σ[idx]*(⊙(𝐠[:,idx.i], 𝐧[:,idx.i])) for idx in idx_set_σ) +
            sum(χ[idx]*(⊙(-𝐠[:,idx.i], 𝐧[:,idx.j] - 𝐧[:,idx.i])) for idx in idx_set_χ) +
            sum(ζ[idx]*(⊙(𝐧[:,idx.i], 𝐧[:,idx.i])) for idx in idx_set_ζ) + 
            - Z,
            SymmetricMatrixShape(dim_Z)
            ) .== 0
        )

    end

end



function solveOptimalStepSizeSmoothRelaxationFW(N, D, L, β;
        functionType = :smooth,                                 # :smooth
        optLoc = :exterior,                                     # :exterior, :inSet
        objectiveType = :finalIterate,                          # :finalIterate, :minIterate
        stepMode = :matrix                                      # :standard, :matrix (corresponds to 'Generalized Frank-Wolfe' where x_{k+1} is a weighted sum of all z_k)
    )

    default_obj_val_upper_bound = 1e6

    μ = 0

    h_test = zeros(N)
    h_test = OffsetArray(h_test,0:N-1)
    
    # Standard step sequence
    for i=0:N-1
        h_test[i] = 2/(i+2)
    end
    h_test[0] = 1
    
    
    dim_Z = 2N+3
    
    ## Solve primal with feasible stepsizes
    
    p_feas, G_feas, Ft_feas, _ = solve_primal_FW_Relaxed(N, h_test, D, L, μ, 0, β, 0; functionType = functionType, setType = :smooth, optLoc = optLoc, objectiveType = objectiveType)
    
    ## -------------------------------------------------------
    # solve the dual for the warm-starting stepsize
    ## -------------------------------------------------------
    
    #NOTE: This function does not change value of h - it just spits it back out.
    d_feas, dualVars_feas, h_feas, idx_sets_feas_eff, model_dual_orig = solve_dual_FW_SmoothRelaxation(
        N, h_test, D, L, μ, β;
        show_output = :off,
        functionType = functionType,
        primalObjective = objectiveType,
        dualObjective = :default,
        optLoc = optLoc,
        obj_val_upper_bound = default_obj_val_upper_bound)
    
    ##  Computing the bounds for computing locally optimal solution to StepDesign-PEP
    
    ##  Using Heuristic bound, comment the block out if using bound based on SDP relaxation
    # -------------------------------------------------------------------------
    
    # Compute M_[]
    M_κ = d_feas/((D-2/β)^2)
    M_ν = d_feas/((D-2/β)^2)
    M_ρ = d_feas/((D-2/β)^2)
    M_ω = d_feas*β^2
    M_ζ = d_feas
    
    M_tilde = 50
    
    # Compute M_λ
    d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(N, h_feas, D, L, μ, β;  show_output = :off,
        functionType = functionType,
        primalObjective = objectiveType,
        dualObjective = :find_M_λ,
        optLoc = optLoc,
        obj_val_upper_bound = 1.001*p_feas)
    
    λ_temp = dualVars_temp[1]
    M_λ = M_tilde*maximum(λ_temp)
    
    #Compute M_η
    d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(N, h_feas, D, L, μ, β;  show_output = :off,
        functionType = functionType,
        primalObjective = objectiveType,
        dualObjective = :find_M_η,
        optLoc = optLoc,
        obj_val_upper_bound = 1.001*p_feas)
    
    η_temp = dualVars_temp[2]
    M_η = M_tilde*maximum(η_temp)
    
    #Compute M_ϕ
    # d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(N, h_feas, D, L, μ, β;  show_output = :off,
    #     functionType = functionType,
    #     primalObjective = objectiveType,
    #     dualObjective = :find_M_ϕ,
    #     optLoc = optLoc,
    #     obj_val_upper_bound = 1.001*p_feas)
    # ϕ_temp = dualVars_temp[3]
    # M_ϕ = M_tilde*maximum(ϕ_temp)

    # Use estimate for M_ϕ
    M_ϕ = M_tilde

    
    #Compute M_σ
    d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(N, h_feas, D, L, μ, β;  show_output = :off,
        functionType = functionType,
        primalObjective = objectiveType,
        dualObjective = :find_M_σ,
        optLoc = optLoc,
        obj_val_upper_bound = 1.001*p_feas)
    
    σ_temp = dualVars_temp[8]
    M_σ = M_tilde*maximum(σ_temp)
    
    #Compute M_χ
    d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(N, h_feas, D, L, μ, β;  show_output = :off,
        functionType = functionType,
        primalObjective = objectiveType,
        dualObjective = :find_M_χ,
        optLoc = optLoc,
        obj_val_upper_bound = 1.001*p_feas)
        
    χ_temp = dualVars_temp[9]
    M_χ = M_tilde*maximum(χ_temp)
    
    # Compute M_Z
    d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(N, h_feas, D, L, μ, β;  show_output = :off,
        functionType = functionType,
        primalObjective = objectiveType,
        dualObjective = :find_M_Z,
        optLoc = optLoc,
        obj_val_upper_bound = 1.001*p_feas)
        
    Z_temp = dualVars_temp[13]
    M_Z = M_tilde*maximum(Z_temp[i,i] for i in 1:dim_Z)
    
    # Compute M_Λ
    
    M_Λ = sqrt(M_Z)

    varBounds = (M_λ, M_η, M_ϕ, M_κ, M_ν, M_ρ, M_ω, M_σ, M_χ, M_ζ, M_Z, M_Λ)
    
    # Optional:
    ## Sparsify the solution for warm-starting locally optimal solver
    # d_temp, dualVars_temp, h_temp, idx_sets_temp_eff, model_dual = solve_dual_FW_SmoothRelaxation(NN, h_feas, D, L, μ, β;  show_output = :off,
    #     functionType = functionType,
    #     primalObjective = objectiveType,
    #     dualObjective = :find_sparse_sol,
    #     obj_val_upper_bound = 1.001*d_feas)
    
    
    
    ## Store the warm start point for computing locally optimal solution
    # d_star_ws, dualVars_ws = d_temp, dualVars_temp
    # idx_sets_ws_eff = idx_sets_temp_eff
    
    d_star_ws, dualVars_ws = d_feas, dualVars_feas
    idx_sets_ws_eff =  idx_sets_feas_eff
    
    
    h_ws = h_temp
    s_ws, ψ_ws = convertHToS(h_ws)
    stepVars_ws = (s_ws, ψ_ws)

    # ---------------------------------------------------
    ## Stage 2 of the StepDesign-PEP Algorithm: compute the locally optimal point
    # ----------------------------------------------------
    obj_val_loc_opt, dualVars_loc_opt, stepVars_loc_opt, idx_sets_loc_opt_eff, model_optStep = localOptStepSolver_SmoothRelaxationFW(
        # different parameters to be used
        # ------------------------------
        N, L, D, β,
        # solution to warm-start
        # ----------------------
        d_star_ws, dualVars_ws, stepVars_ws,
        # index sets
        # -----------------------
        idx_sets_ws_eff,
        # bounds on the variables
        # ----------------------
        varBounds,
        # Required options
        # ----------------------
        functionType,
        objectiveType,
        optLoc,
        stepMode;
        # options
        # -------
        show_output = :on, # other option :on
        reduce_index_sets = :for_warm_start_only,
        bound_impose = :off, # if this is :on, then from the warm_start solution we compute lower and upper bounds for the decision variables using the semidefinite relaxation
        quadratic_equality_modeling = :exact,
        cholesky_modeling = :definition,
        ϵ_tol_feas = 1e-6, # tolerance for Cholesky decomposition,
        polish_solution = :off, # wheather to polish the solution to get better precision, the other option is :off
        print_model=:off
    )

    (s_loc_opt, q_loc_opt, QQ_loc_opt, ψ_loc_opt) = stepVars_loc_opt
    
    # Convert s variable to h variable for interpretability
    if s_loc_opt == -1
        h_loc_opt = -1
    else
        if stepMode == :matrix
            h_loc_opt = convertSToHH(s_loc_opt; inputType = :variable)
        else 
            h_loc_opt = convertSToH(s_loc_opt; inputType = :variable)
        end
    end

    return obj_val_loc_opt, h_loc_opt

end


#Special index set constructor for QQ placeholder variable
function index_set_constructor_QQ(N)

    #construct the index set for dual variable
    idx_set = ijk_idx[]
    for i in 0:N-1        # We can handle i=-1 and i=0 normally, so skip them in this formulation
        for j in 0:N-1
            if j>i    # We only need j indices up to i
                continue
            end
            for k in 0:N-1
                if (k>i)||(k>j) # We only need k indices up to i
                    continue        # And within each i, make it lower triangular wrt k,j since qij*qik = qik*qij
                end
                push!(idx_set, ijk_idx(i,j,k))
            end
        end
    end

    return idx_set
end


# Special function for QQ handling for ω constraint with ||x_i - w_i || <= 1/β
specialMatQQ(QQ,q,i,𝐳,𝐰) = sum( QQ[ijk_idx(i-1,max(j,k),min(j,k))]*(⊙(𝐳[:,j], 𝐳[:,k])) for j in 0:i-1, k in 0:i-1) +
                            -2*sum(q[ij_idx(i-1,j)]*(⊙(𝐳[:,j], 𝐰[:,i])) for j in 0:i-1) +
                            ⊙(𝐰[:,i], 𝐰[:,i])

#Note: We use max, min because QQ is lower triangular, but we should still include cases k>j, since those should be double-counted
