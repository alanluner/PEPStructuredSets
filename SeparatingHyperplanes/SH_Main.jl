using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools

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


# Solve PEP for separating hyperplane problem
function solve_primal_SH(N, R, h, α, β, δ,
        setType;                                    # :convex, :smooth, :SC 
        modelOnly = false                           # For feasibility debugging, set to true to return model without solving
    )

    # Generate selection vectors
    if setType == :convex
        𝐱, 𝐳, 𝐧, 𝐪, 𝐰 = data_generator_SH_Convex(N,h)
        dim_G = 2N+3
    elseif setType == :smooth
        𝐱, 𝐳, 𝐧, 𝐪, 𝐰 = data_generator_SH_Smooth(N,h)
        dim_G = 2N+4
    elseif setType == :SC
        𝐱, 𝐳, 𝐧, 𝐪, 𝐰 = data_generator_SH_SC(N,h)
        dim_G = 2N+3
    end

    # number of points etc
    # --------------------

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

    #Set objective 
    @objective(model, Max, tr(G*(⊙(𝐧[:,N], 𝐱[:,N] - 𝐳[:,N]))))

    # Apply set constraints
    applySHConstraints(model, setType, N, R, α, β, δ, G, 𝐱, 𝐳, 𝐧, 𝐪, 𝐰)


    # time to optimize
    # ----------------

    if modelOnly
        return -1, -1, -1, model, 𝐱, 𝐳, 𝐧, 𝐪, 𝐰
    end

    set_silent(model)

    optimize!(model)

    # store and return the solution
    # -----------------------------

    discard = primal_status(model) != MOI.FEASIBLE_POINT
    if discard
        # print(primal_status(model))
        # print(termination_status(model))
    end

    p_star = objective_value(model)

    G_star = value.(G)

    return p_star, G_star, discard, model, 𝐱, 𝐳, 𝐧, 𝐪, 𝐰

end


# Generate selection vectors for G, for convex sets
function data_generator_SH_Convex(N, h; variable_h = :off)

    dim_G = 2N+3

    𝐱_0 = e_i(dim_G, 1)

    # 𝐳 = [𝐳_0  𝐳_1 ... 𝐳_N] 
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, 0:N)
    for i in 0:N
        𝐳[:,i] = e_i(dim_G, i+2)
    end

    # 𝐧 = [𝐧_0  𝐧_1 ... 𝐧_N] 
    𝐧 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, 0:N)
    for i in 0:N
        𝐧[:,i] = e_i(dim_G, i+(N+3))
    end

    # 𝐱= [𝐱_0 𝐱_1 𝐱_2 ... 𝐱_N]
    if variable_h == :off
        𝐱 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, 0:N)
        𝐱[:,0] = 𝐱_0
        for k in 1:N
            𝐱[:,k] = 𝐱[:,k-1] - h[k-1]*𝐧[:,k-1]
        end
    else
        𝐱 = 𝐱_0
        for k=1:N
            𝐱_k = 𝐱_0 - sum( h[i] .* 𝐧[:,i] for i in 0:k-1)
            𝐱 = [𝐱   𝐱_k]
        end
        
        #NOTE: Offset has to happen last. Appending will mess it up
        𝐱 = OffsetArray(𝐱, 1:dim_G, 0:N)
    end

    𝐪 = zeros(dim_G)
    𝐰 = -1

    return 𝐱, 𝐳, 𝐧, 𝐪, 𝐰

end

# Generate selection vectors for G, for SC sets
function data_generator_SH_SC(N,h; variable_h = :off)
    return data_generator_SH_Convex(N,h; variable_h = variable_h)
end

# Generate selection vectors for G, for smooth sets
function data_generator_SH_Smooth(N,h; variable_h = :off)
    dim_G = 2N+4

    𝐱_0 = e_i(dim_G, 1)

    # 𝐳 = [𝐳_0  𝐳_1 ... 𝐳_N] 
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, 0:N)
    for i in 0:N
        𝐳[:,i] = e_i(dim_G, i+2)
    end

    # 𝐧 = [𝐧_0  𝐧_1 ... 𝐧_N] 
    𝐧 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, 0:N)
    for i in 0:N
        𝐧[:,i] = e_i(dim_G, i+(N+3))
    end

    𝐰 = e_i(dim_G, 2N+4)

    # 𝐱= [𝐱_0 𝐱_1 𝐱_2 ... 𝐱_N]
    if variable_h == :off
        𝐱 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, 0:N)
        𝐱[:,0] = 𝐱_0
        for k in 1:N
            𝐱[:,k] = 𝐱[:,k-1] - h[k-1]*𝐧[:,k-1]
        end
    else
        𝐱 = 𝐱_0
        for k=1:N
            𝐱_k = 𝐱_0 - sum( h[i] .* 𝐧[:,i] for i in 0:k-1)
            𝐱 = [𝐱   𝐱_k]
        end
        
        #NOTE: Offset has to happen last. Appending will mess it up
        𝐱 = OffsetArray(𝐱, 1:dim_G, 0:N)
    end

    𝐪 = zeros(dim_G)

    return 𝐱, 𝐳, 𝐧, 𝐪, 𝐰

end

# Apply set constraints
function applySHConstraints(model, setType, N, R, α, β, δ, G, 𝐱, 𝐳, 𝐧, 𝐪, 𝐰)
    if setType == :convex
        applySHConstraints_Convex(model, N, R, δ, G, 𝐱, 𝐳, 𝐧, 𝐪)
    elseif setType == :smooth
        applySHConstraints_Smooth(model, N, R, β, δ, G, 𝐱, 𝐳, 𝐧, 𝐪, 𝐰)
    elseif setType == :SC
        applySHConstraints_SC(model, N, R, α, δ, G, 𝐱, 𝐳, 𝐧, 𝐪)
    end
end


function applySHConstraints_Convex(model, N, R, δ, G, 𝐱, 𝐳, 𝐧, 𝐪)

    I = 0:N

    𝐱_0 = 𝐱[:,0]

    #[1] ( ||x_0 - x_⋆|| <= R )
    @constraint(model, tr(G*(⊙(𝐱_0, 𝐱_0))) <= R^2)

    for i in I
        for j in I
            if i==j
                continue
            end
            #[2] (Interp1)
            @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐳[:,j] - 𝐳[:,i]))) <= 0)

        end
    end

    for i in I
        #[3] (Interp2)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐪 + δ*𝐧[:,i] - 𝐳[:,i]))) <= 0)

        #[4] (n_i is separating hyperplane)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐳[:,i] - 𝐱[:,i]))) <= 0)

        #[5] (n_i is unit)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐧[:,i]))) == 1)
    end

end


function applySHConstraints_SC(model, N, R, α, δ, G, 𝐱, 𝐳, 𝐧, 𝐪)

    I = 0:N

    𝐱_0 = 𝐱[:,0]

    #[1] ( ||x_0 - x_⋆|| <= R )
    @constraint(model, tr(G*(⊙(𝐱_0, 𝐱_0))) <= R^2)

    for i in I
        for j in I
            if i==j
                continue
            end
            #[2] (Interp1)
            @constraint(model, tr(G*(⊙(𝐳[:,j] - 𝐳[:,i] + 1/α*𝐧[:,i], 𝐳[:,j] - 𝐳[:,i] + 1/α*𝐧[:,i]))) <= 1/α^2)

        end
    end

    for i in I
        #[3] (Interp2)
        @constraint(model, tr(G*(⊙(𝐪 - 𝐳[:,i] + 1/α*𝐧[:,i], 𝐪 - 𝐳[:,i] + 1/α*𝐧[:,i]))) <= (1/α-δ)^2)

        #[4] (n_i is separating hyperplane)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐳[:,i] - 𝐱[:,i]))) <= 0)

        #[5] (n_i is unit)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐧[:,i]))) == 1)
    end

end


function applySHConstraints_Smooth(model, N, R, β, δ, G, 𝐱, 𝐳, 𝐧, 𝐪, 𝐰)
    
    I = 0:N

    𝐱_0 = 𝐱[:,0]

    s = max(0, δ-1/β)

    #[1] ( ||x_0 - x_⋆|| <= R )
    @constraint(model, tr(G*(⊙(𝐱_0, 𝐱_0))) <= R^2)

    for i in I
        for j in I
            if i==j
                continue
            end
            #[2] (Interp1)
            @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐳[:,j] - 1/β*𝐧[:,j] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= 0)

        end
    end

    for i in I
        #[3] (Interp2)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐰 + s*𝐧[:,i] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= 0)

        #[4] (n_i is separating hyperplane))
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐳[:,i] - 𝐱[:,i]))) <= 0)

        #[5] (n_i is unit)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐧[:,i]))) == 1)
    end

    #[6] (Interp3)
    @constraint(model, tr(G*(⊙(𝐪 - 𝐰, 𝐪 - 𝐰))) <= (1/β - δ + s)^2 )

end


# Solve Separating Hyperplane halting problem (Find NMax that guarantees x_N ∈ int C), with an initial guess (to reduce computation cost)
function solve_SH_HaltingProblem_WithGuess(R, α, β, δ, 
        setType,                                            # :convex, :smooth, :SC
        NMaxGuess                                           # Initial guess at NMax
    )

    #Verify that the problem succeeds for NMaxGuess - 1 and fails at NMaxGuess

    NInit = NMaxGuess - 1

    NMax = solve_SH_HaltingProblem(R, α, β, δ, setType; NInit = NInit, NUpperLimit = NMaxGuess + 5)

    if NMax == NMaxGuess
        # print("Correct guess NMax=", NMax)
        return NMax
    else
        print("Incorrect guess NMax=", NMax)
        return NMax
    end

end

# Solve Separating Hyperplane halting problem (Find NMax that guarantees x_N ∈ int C)
function solve_SH_HaltingProblem(R, α, β, δ, setType; NInit = 1, NUpperLimit = 50)

    #First we verify that problem succeeds for NInit
    h = max(δ, 1/β)*ones(NInit)
    h = OffsetArray(h, 0:NInit-1)

    p, _, discard, _ = solve_primal_SH(NInit, R, h, α, β, δ, setType)
    if (p < 1e-6)||discard
        @warn "Problem infeasible for" NInit ". Use a smaller starting value."
        return -1
    end

    global NMax = -1

    # Then run with increasing N until it fails. The first N for which it fails is NMax
    for N in NInit+1:NUpperLimit

        h = max(δ, 1/β)*ones(N)
        h = OffsetArray(h, 0:N-1)

        p, _, discard, _ = solve_primal_SH(N, R, h, α, β, δ, setType)

        # If failure, return this N as NMax
        if (p < 1e-6)||discard
            NMax = N 
            break
        end
    end

    return NMax

end