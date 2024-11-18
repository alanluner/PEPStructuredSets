using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, KNITRO, Optim


# construct e_i in R^n
function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

# symmetric outer product
function ⊙(a,b)
    return ((a*b') .+ transpose(a*b')) ./ 2
end

# General solver for different versions/subproblems of Alternating Projections PEP
function solve_primal_AP_General(N, R, α, δ, 
        setType,                                    # :convex, :SC (:smooth not supported)
        mode;                                       # :cut, :onCurve
        divU=0, divV=0,                             # If mode == :cut, partition values for ||u||^2 and ||v||^2
        uSqVals = 0, vSqVals = 0,                   # If mode == :onCurve, fixed values for ||u||^2 and ||v||^2
        modelOnly=false                             # For feasibility debugging, set to true to return model without solving
    )

    # Generate selection vectors
    if setType == :convex
        𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪 = data_generator_AP_Convex(N)
        dim_G = 4N+6
    elseif setType == :smooth
        @warn "setType smooth not supported for AP"
    elseif setType == :SC
        𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪 = data_generator_AP_SC(N)
        dim_G = 4N+6
    end

    𝐲 = zeros(dim_G)

    # define index sets
    # --------------------

    I = vcat(-1,0:N-1)
    K = vcat(-1,1:N)

    # define the model
    # ----------------

    model = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model, G[1:dim_G, 1:dim_G])
    if !modelOnly #If we are debugging and only want the model output, skip PSD constraint since feasibility report can't handle it
        @constraint(model, psd, G >= 0, PSDCone() )
    end
    
    # define objective
    # ----------------

    #Set objective based on problem type
    @objective(model, Max, tr(G*(⊙(𝐱[:,N] - 𝐲, 𝐱[:,N] - 𝐲))))

    # Apply set constraints
    applyAPConstraints(model, setType, N, R, α, δ, G, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪)

    # Branch behavior depending on :mode
    if mode == :cut
        applyCuts_AP(model, divU, divV, G, N, 𝐮, 𝐯, 𝐦, 𝐧)
    elseif mode == :onCurve
        forceToCurve_AP(model, G, N, uSqVals, vSqVals, 𝐮, 𝐯, 𝐦, 𝐧)
    end

    # optimize
    # ----------------

    if modelOnly
        return -1, -1, -1, model, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪
    end

    set_silent(model)

    optimize!(model)

    # store and return the solution
    # -----------------------------

    discard = primal_status(model) != MOI.FEASIBLE_POINT
    if discard
        # print(primal_status(model))
    end

    p_star = objective_value(model)

    G_star = value.(G)

    return p_star, G_star, discard, model, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪

end


# Solve ̃p subproblem with slice over ||u||^2 and ||v||^2 values
function solve_primal_AP_WithCut(N, R, α, δ, setType, divU, divV; modelOnly=false)
    mode = :cut
    return solve_primal_AP_General(N, R, α, δ, setType, mode; divU = divU, divV = divV, modelOnly = modelOnly)
end

# Solve ̂p subproblem with fixed ||u||^2 and ||v||^2 values
function solve_primal_AP_OnCurve(N, R, α, δ, setType, uSqVals, vSqVals; modelOnly = false)
    mode = :onCurve
    return solve_primal_AP_General(N, R, α, δ, setType, mode; uSqVals = uSqVals, vSqVals = vSqVals, modelOnly = modelOnly )
end

# Solve p_relaxed SDP (relaxation to get upper bound)
function solve_primal_AP_NoCut(N, R, α, δ, setType; modelOnly=false)
    mode = :noCut
    return solve_primal_AP_General(N, R, α, δ, setType, mode; modelOnly = modelOnly)
end

# Generate selection vectors for G, for convex sets
function data_generator_AP_Convex(N)

    dim_G = 4N+6 

    𝐱_0 = e_i(dim_G, 1)

    zeroVec = zeros(dim_G)

    # 𝐮 = [𝐮_⋆ 𝐮_0 𝐮_1 ... 𝐮_N-1]
    𝐮 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in -1:N-1
        𝐮[:,i] = e_i(dim_G, i+3)
    end

    # 𝐯 = [𝐯_⋆ [𝐯_0 (*ignored*)]  𝐯_1 ... 𝐯_N] 
    𝐯 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    𝐯[:,-1] = e_i(dim_G, N+3)
    #Skip v_0 since it does not exist
    for i in 1:N
        𝐯[:,i] = e_i(dim_G, i+(N+3))
    end

    # 𝐦 = [𝐦_⋆ 𝐦_0 𝐦_1 ... 𝐦_N-1]
    𝐦 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in -1:N-1
        𝐦[:,i] = e_i(dim_G, i+(2N+5))
    end

    # 𝐧 = [𝐧_⋆ [𝐧_0 (*ignored*)]  𝐧_1 ... 𝐧_N] 
    𝐧 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    𝐧[:,-1] = e_i(dim_G, 3N+5)
    #Skip v_0 since it does not exist
    for i in 1:N
        𝐧[:,i] = e_i(dim_G, i+(3N+5))
    end

    𝐪 = e_i(dim_G, 4N+6)

    # 𝐱= [𝐱_⋆ 𝐱_0 𝐱_1 𝐱_2 ... 𝐱_N]
    𝐱 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    𝐱[:,0] = 𝐱_0
    for k in 1:N
        𝐱[:,k] = 𝐱_0 - sum(𝐮[:,j] for j in 0:k-1; init=zeroVec) - sum(𝐯[:,j] for j in 1:k; init=zeroVec)
    end

    # 𝐳 = [𝐳_⋆ 𝐳_0 𝐳_1 𝐳_2 ... 𝐳_N-1]
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in 0:N-1
        𝐳[:,i] = 𝐱_0 - sum(𝐮[:,j] for j in 0:i; init=zeroVec) - sum(𝐯[:,j] for j in 1:i; init=zeroVec)
    end

    return 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪

end

# Generate selection vectors for G, for strongly convex sets
function data_generator_AP_SC(N)
    return data_generator_AP_Convex(N)
end

# Apply set constraints to model
function applyAPConstraints(model, setType, N, R, α, δ, G, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪)
    if setType == :convex
        applyAPConstraints_Convex(model, N, R, δ, G, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪)
    elseif setType == :SC
        applyAPConstraints_SC(model, N, R, α, δ, G, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪)
    end
end

# Constraints for convex sets
function applyAPConstraints_Convex(model, N, R, δ, G, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪)
    I = -1:N-1
    K = vcat(-1,1:N)
    
    𝐲 = 𝐱[:,-1]

    for i in I
        for j in I
            if i==j
                continue
            end
            #[1] (Interp1)
            @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐳[:,j] - 𝐳[:,i]))) <= 0)

            #[2] (Relaxation of m = u/||u||)
            @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,j] - 𝐦[:,i]))) <= 0)
        end
    end

    for k in K
        for l in K
            if k==l
                continue
            end
            #[3] (Interp1)
            @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐱[:,l] - 𝐱[:,k]))) <= 0)

            #[4] (Relaxation of n = v/||v||)
            @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,l] - 𝐧[:,k]))) <= 0)
        end
    end

    for i in I
        #[5] (Interp2)
        @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐪 - 𝐳[:,i] + δ*𝐦[:,i]))) <= 0)

        #[6] (Relaxation of m = u/||u||)
        @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,i]))) >= 0)

        #[7] (Relaxation of m = u/||u||)
        @constraint(model, tr(G*(⊙(𝐦[:,i], 𝐦[:,i]))) == 1)
    end

    for k in K
        #[8] (Interp2)
        @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐪 - 𝐱[:,k] + δ*𝐧[:,k]))) <= 0)

        #[9] (Relaxation of n = v/||v||)
        @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,k]))) >= 0)

        #[10] (Relaxation of n = v/||v||)
        @constraint(model, tr(G*(⊙(𝐧[:,k],𝐧[:,k]))) == 1)
    end

    #[11] x_N = y + u_⋆ + v_⋆
    @constraint(model, tr(G*(⊙(𝐱[:,N] - 𝐲 - 𝐮[:,-1] - 𝐯[:,-1],  𝐱[:,N] - 𝐲 - 𝐮[:,-1] - 𝐯[:,-1]))) == 0)

    #[12] ||x_0 - q|| <= R
    @constraint(model, tr(G*(⊙(𝐱[:,0] - 𝐪, 𝐱[:,0] - 𝐪))) <= R^2)

end

# Constraints for strongly convex sets
function applyAPConstraints_SC(model, N, R, α, δ, G, 𝐱, 𝐳, 𝐮, 𝐯, 𝐦, 𝐧, 𝐪)
    I = -1:N-1
    K = vcat(-1,1:N)
    
    𝐲 = 𝐱[:,-1]

    for i in I
        for j in I
            if i==j
                continue
            end
            #[1] (Interp1)
            @constraint(model, tr(G*(⊙(𝐳[:,i] - 𝐳[:,j] + 1/α*𝐦[:,j], 𝐳[:,i] - 𝐳[:,j] + 1/α*𝐦[:,j]))) <= 1/α^2)

            #[2] (Relaxation of m = u/||u||))
            @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,j] - 𝐦[:,i]))) <= 0)

        end
    end

    for k in K
        for l in K
            if k==l
                continue
            end
            #[3] (Interp1)
            @constraint(model, tr(G*(⊙(𝐱[:,k] - 𝐱[:,l] + 1/α*𝐧[:,l],  𝐱[:,k] - 𝐱[:,l] + 1/α*𝐧[:,l]))) <= 1/α^2)

            #[4] (Relaxation of n = v/||v||)
            @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,l] - 𝐧[:,k]))) <= 0)
        end
    end

    for i in I
        #[5] (Interp2)
        @constraint(model, tr(G*(⊙(𝐪 - 𝐳[:,i] + 1/α*𝐦[:,i],  𝐪 - 𝐳[:,i] + 1/α*𝐦[:,i]))) <= (1/α - δ)^2)

        #[6] (Relaxation of m = u/||u||)
        @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,i]))) >= 0)

        #[7] (Relaxation of m = u/||u||)
        @constraint(model, tr(G*(⊙(𝐦[:,i], 𝐦[:,i]))) == 1)
    end

    for k in K
        #[8] (Interp2)
        @constraint(model, tr(G*(⊙(𝐪 - 𝐱[:,k] + 1/α*𝐧[:,k],   𝐪 - 𝐱[:,k] + 1/α*𝐧[:,k]))) <= (1/α - δ)^2)

        #[9] (Relaxation of n = v/||v||)
        @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,k]))) >= 0)

        #[10] (Relaxation of n = v/||v||)
        @constraint(model, tr(G*(⊙(𝐧[:,k], 𝐧[:,k]))) == 1)
    end

    #[11] x_N = y + u_⋆ + v_⋆
    @constraint(model, tr(G*(⊙(𝐱[:,N] - 𝐲 - 𝐮[:,-1] - 𝐯[:,-1],  𝐱[:,N] - 𝐲 - 𝐮[:,-1] - 𝐯[:,-1]))) == 0)

    #[12] ||x_0 - q|| <= R
    @constraint(model, tr(G*(⊙(𝐱[:,0] - 𝐪, 𝐱[:,0] - 𝐪))) <= R^2)

end

# Apply cuts to solve SDP over convex subregion (Section 5.4.3 of paper)
function applyCuts_AP(model, divU, divV, G, N, 𝐮, 𝐯, 𝐦, 𝐧)
    I = -1:N-1
    K = vcat(-1, 1:N)

    for i in I
        divU1 = divU[i,1]
        divU2 = divU[i,2]

        if divU2 == -1
            # If divU2 == -1, then this is the terminal cut so it should be horizontal
            @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,i]))) >= sqrt(divU1) )
        else
            #Otherwise, do a normal cut
            @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,i] - 1/(sqrt(divU1) + sqrt(divU2))*𝐮[:,i]))) >= sqrt(divU1*divU2)/(sqrt(divU1) + sqrt(divU2)) )
        end
    end

    for k in K
        divV1 = divV[k,1]
        divV2 = divV[k,2]

        if divV2 == -1
            # If divU2 == -1, then this is the terminal cut so it should be horizontal
            @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,k]))) >= sqrt(divV1) )
        else
            #Otherwise, do a normal cut
            @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,k] - 1/(sqrt(divV1) + sqrt(divV2))*𝐯[:,k]))) >= sqrt(divV1*divV2)/(sqrt(divV1) + sqrt(divV2)) )
        end
    end
end

# Initialize branch using our chosen upper bound on ||g||
function getInitialBranch(valMax,N)
    branch = zeros(N+1,2)
    branch = OffsetArray(branch, -1:N-1, 1:2)
    branch[:,1] .= 0
    branch[:,2] .= valMax

    return branch
end

# Apply constraints to fix ||u_i||^2 and <u_i, m_i> (and similar for n,v)
function forceToCurve_AP(model, G, N, uSqVals, vSqVals, 𝐮, 𝐯, 𝐦, 𝐧)

    #Include abs to handle numerical errors
    umVals = sqrt.(abs.(uSqVals))
    vnVals = sqrt.(abs.(vSqVals))

    for i=-1:N-1
        @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐮[:,i]))) == uSqVals[i] )
        @constraint(model, tr(G*(⊙(𝐮[:,i], 𝐦[:,i]))) == umVals[i] )
    end

    K = vcat(-1,1:N)
    for k in K
        @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐯[:,k]))) == vSqVals[k] )
        @constraint(model, tr(G*(⊙(𝐯[:,k], 𝐧[:,k]))) == vnVals[k] )
    end
end

# Approximate solution for convex sets, based on conjecture 6.2 of exact rate
function getAPLowerBound(N,R,δ)

    ff(c) = -(c[1]^(2*N-1)*sqrt(R^2-δ^2) - δ*c[1]^(2*N)/sqrt(1 - c[1]^2) - δ*c[1]^(2*N-1)/sqrt(1 - c[1]^2))

    c0 = 0.5*ones(1)
    lower = 0.0
    upper = 1.0
    
    #Need fminbox here or it doesn't work
    
    sol = Optim.optimize(ff, lower, upper, c0, Fminbox(NelderMead()), Optim.Options(show_trace = false, show_every = 50, g_tol = 1e-5))

    result = Optim.minimum(sol)^2 #We square it because SDP uses fN^2
    cOpt = Optim.minimizer(sol)

    return result, cOpt

end