## Load the packages:
# -------------------
using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, KNITRO

include("FW_Branching.jl")


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



A_mat(i,j,𝐠,𝐱) = ⊙(𝐠[:,j], 𝐱[:,i]-𝐱[:,j])
B_mat(i,j,𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])
C_mat(i,j,𝐠) = ⊙(𝐠[:,i]-𝐠[:,j], 𝐠[:,i]-𝐠[:,j])
a_vec(i,j,𝐟) = 𝐟[:, j] - 𝐟[:, i]


# General solver for different versions/subproblems of Frank-Wolfe PEP 
function solve_primal_FW_General(N, h, D, L, μ, α, β, δ, 
        mode;                                                   # :cut, :onCurve
        functionType = :smooth,                                 # :smooth, :smoothSC
        setType = :convex,                                      # :convex, :smooth, :SC, :smoothSC
        objectiveType = :minIterate,                            # :minIterate, :finalIterate
        optLoc = :exterior,                                     # :exterior, :inSet, :interior
        modelOnly = false,                                      # For feasibility debugging, set to true to return model without solving 
        divVals = 0,                                            # If mode == :cut, partition values for ||g_i||^2
        gSqVals = 0,                                            # If mode == :onCurve, fixed values for ||g_i||^2
        SCDiam = :none                                          # :large, :small - Determines choice of λ for diameter constraints. :large corresponds to λ=1 and results in an upper bound for the true PEP result. :small corresponds to λ=1/sqrt(2) and results in a lower bound for the true PEP result.
    )

    # Generate selection vectors
    if setType == :convex
        if optLoc == :interior
            𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetConvex_Interior(N, h)
            dim_G = 3N+4
        else
            𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetConvex(N, h)
            dim_G = 2N+3
        end
    elseif setType == :smooth
        𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetSmooth(N, h)
        dim_G = 4N+6
    elseif setType == :SC
        𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetSC(N, h)
        dim_G = 3N+4
    elseif setType == :smoothSC
        𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟 = data_generator_SetSmoothSC(N, h)
        dim_G = 4N+6
    else
        @warn "Invalid setType"
    end

    # define index sets
    # --------------------

    I = -1:N-1
    K = -1:N

    dim_Ft = N+1

    # define the model
    # ----------------

    model = Model(optimizer_with_attributes(Mosek.Optimizer))

    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model, G[1:dim_G, 1:dim_G])
    if !modelOnly
        @constraint(model, psd, G >=0, PSDCone())
    end

    # construct Ft (transpose of F)
    @variable(model, Ft[1:dim_Ft])


    # define objective
    # ----------------

    #Set objective based on problem type
    if objectiveType == :finalIterate
        @objective(model, Max, Ft'*a_vec(-1,N,𝐟))
    else
        # If objective is minIterate, define new variable t to represent minIterate
        @variable(model, t)
        @objective(model, Max, t)
        for i=0:N
            @constraint(model, t <= Ft'*a_vec(-1,i,𝐟))
        end
    end

    # constraints
    # ------------------------

    # function interpolation constraints
    setPrimalFuncConstraints(model, functionType, Ft, G, 𝐟, 𝐠, 𝐱, h, K, L, μ)

    # set interpolation constraints
    applySetConstraints(model, setType, N, D, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, α, β, δ; optLoc = optLoc, SCDiam = SCDiam)

    # Branch behavior depending on :mode
    if mode == :cut
        applyCuts(model, divVals, G, N, 𝐠, 𝐧)
    elseif mode == :onCurve
        forceToCurve_FW(model, G, N, gSqVals, 𝐠, 𝐧, optLoc)
    end


    # optimize
    # ----------------

    if modelOnly
        return -1, -1, -1,-1,model, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟,-1
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

    Ft_star = value.(Ft)

    # Check if final iterate is the unique minimum
    isFinalIterateBest = minimum(Ft_star[1:N]) >= (1.01)*Ft_star[N+1]

    return p_star, G_star, Ft_star, discard, model, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟, isFinalIterateBest

end

# Solve ̃p subproblem with slice over ||g_i||^2 values
function solve_primal_FW_WithCut(N, h, D, L, μ, α, β, δ, divVals; functionType = :smooth, setType = :convex, objectiveType = :minIterate, optLoc = :exterior, modelOnly = false, SCDiam = :none)
    mode = :cut
    return solve_primal_FW_General(N, h, D, L, μ, α, β, δ, mode; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = modelOnly, divVals = divVals, SCDiam = SCDiam)
end

# Solve ̂p subproblem with fixed ||g_i||^2 values
function solve_primal_FW_OnCurve(N, h, D, L, μ, α, β, δ, gSqVals; functionType = :smooth, setType = :convex, objectiveType = :minIterate, optLoc = :exterior, modelOnly = false, SCDiam = :none)
    mode = :onCurve
    return solve_primal_FW_General(N, h, D, L, μ, α, β, δ, mode; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = modelOnly, gSqVals = gSqVals, SCDiam = SCDiam)
end

# Solve p_relaxed SDP (relaxation to get upper bound)
function solve_primal_FW_Relaxed(N, h, D, L, μ, α, β, δ; functionType = :smooth, setType = :convex, objectiveType = :minIterate, optLoc = :exterior, modelOnly = false, SCDiam = :none)
    mode = :noCut
    return solve_primal_FW_General(N, h, D, L, μ, α, β, δ, mode; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = modelOnly, SCDiam = SCDiam)
end


# Special case for convex sets, and optLoc = :exterior or :inSet
function solve_primal_FW_Convex(N, h, D, L, μ; optLoc = :exterior, functionType = :smooth, objectiveType = :minIterate, modelOnly = false)
    mode = :noCut
    setType = :convex
    return solve_primal_FW_General(N, h, D, L, μ, 0, 0, 0, mode; functionType = functionType, setType = setType, optLoc = optLoc, objectiveType = objectiveType, modelOnly = modelOnly)
end
    
# Apply cuts to solve SDP over convex subregion (Section 5.4.3 of paper)
function applyCuts(model, divVals, G, N, 𝐠, 𝐧)
    for i=-1:N-1
        div1 = divVals[i,1]
        div2 = divVals[i,2]

        if div2 == -1
            #If div2=-1, then this is the terminal cut, so it should be horizontal
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐧[:,i]))) >= sqrt(div1) )
        else
            #Otherwise, do a normal cut
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐧[:,i] + 1/(sqrt(div1) + sqrt(div2))*𝐠[:,i]))) >= sqrt(div1*div2)/(sqrt(div1) + sqrt(div2)) )
        end
    end
end


# Apply constraints to fix ||g_i||^2 and <-g_i, n_i>
function forceToCurve_FW(model, G, N, gSqVals, 𝐠, 𝐧, optLoc)

    #Include abs to handle numerical errors
    gnVals = sqrt.(abs.(gSqVals)) #Add abs to catch numerical errors around zero

    #Apply exact constraints #Note: these vars are already offset arrays
    for i=-1:N-1
        if (optLoc in [:inSet, :interior])&&(i==-1)
            continue
        end
        @constraint(model, tr(G*(⊙(𝐠[:,i], 𝐠[:,i]))) == gSqVals[i])
        @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐧[:,i]))) == gnVals[i])
    end

end

# Wrapper function for solve_primal_FW_OnCurve
# Purpose is to pass gSqVals as a standard vector (not offset), so that Optim can handle it
function solve_primal_FW_OnCurve_VecWrapper(N, h, D, L, μ, α, β, δ, p; functionType = :smooth, setType = :smooth, objectiveType = :finalIterate, optLoc = :exterior, modelOnly = false, SCDiam = :none)
    if optLoc in [:inSet, :interior]
        p = vcat(0,p)
    end
    gSqVals = OffsetArray(p, -1:N-1)
    return solve_primal_FW_OnCurve(N, h, D, L, μ, α, β, δ, gSqVals; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = false, SCDiam = SCDiam)
end