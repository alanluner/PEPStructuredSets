
using JuMP, MosekTools, Mosek, LinearAlgebra,  OffsetArrays,  Gurobi, Ipopt, JLD2, Distributions, OrderedCollections, BenchmarkTools, Optim

#####-----------------------------#####
# Some of this code is adapted from the Branch-And-Bound PEP solver of Shuvomoy Das Gupta, Bart P.G. Van Parys, and Ernest K. Ryu.
# Accessible at https://github.com/Shuvomoy/BnB-PEP-code
# 
# Shuvomoy Das Gupta, Bart P.G. Van Parys, Ernest K. Ryu, "Branch-and-Bound Performance Estimation Programming: A Unified Methodology for Constructing Optimal Optimization Methods",
# Mathematical Programming 204.1 (2024): 567-639.
#####-----------------------------#####


# construct e_i in R^n
function e_i(n, i)
    e_i_vec = zeros(n, 1)
    e_i_vec[i] = 1
    return e_i_vec
end

# this symmetric outer product
function ⊙(a,b)
    return ((a*b') .+ transpose(a*b')) ./ 2
end


B_mat(i,j,h,𝐱) = ⊙(𝐱[:,i]-𝐱[:,j], 𝐱[:,i]-𝐱[:,j])


# Generate selection vectors for G (and F)
function data_generator_EpiSmooth(N, h, β;
    momentum=false                              # If true, h is ignored and (modified) Nesterov momentum is used
)

    dim_G = N+2
    dim_F = 2N+3
    
    # Construct 𝐱_0, 𝐱_i and 𝐛_i with respect to G

    # define 𝐱_0 and 𝐱_star
    𝐱_0 = e_i(dim_G, 1)

    𝐱_star = zeros(dim_G, 1)

    # 𝐛 = [𝐛_⋆ 𝐛_0 𝐛_1 𝐛_2 ... 𝐛_N]
    𝐛 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in 0:N        #Skip -1 to enforce 𝐛_⋆ = 0
        𝐛[:,i] = e_i(dim_G, i+2)
    end
    
    # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}]
    𝐱 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    
    if momentum
        zeta = generateMomentumWeights_Nesterov(N)

        𝐱[:,0] = 𝐱_0
        𝐱[:,1] = 𝐱[:,0] - (1/β).*𝐛[:,1]
    
        for i in 1:N-1
            𝐱[:,i+1] = 𝐱[:,i] - (1/β).*𝐛[:,i] + (zeta[i]-1)/zeta[i+1]*( (𝐱[:,i] - (1/β).*𝐛[:,i]) - (𝐱[:,i-1] - (1/β).*𝐛[:,i-1]) )
        end
    else
        𝐱[:,0] = 𝐱_0

        for k in 1:N
            𝐱[:,k] = 𝐱[:,k-1] - h[k-1]/β*𝐛[:,k-1]
        end
    end

    # Construct 𝐟 and 𝐭

    # 𝐟 = [𝐟_⋆ 𝐟_0 𝐟_1 ... 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_F, N+2), 1:dim_F, -1:N)
    for i in 0:N        # Skip -1 to enforce 𝐟_⋆ = 0
        𝐟[:,i] = e_i(dim_F, i+1)
    end
    
    # 𝐭 = [𝐭_⋆ 𝐭_0 𝐭_1 ... 𝐭_N]

    𝐭 = OffsetArray(zeros(dim_F, N+2), 1:dim_F, -1:N)
    for i in -1:N        # Don't skip -1
        𝐭[:,i] = e_i(dim_F, i+(N+3))
    end

    return 𝐱, 𝐛, 𝐟, 𝐭

end


# Solve PEP for epismooth gradient method
function solvePrimalEpiSmooth(N, R, β, h;
        useMomentum = false,                    # If true, h is ignored and (modified) Nesterov momentum is used
        modelOnly = false,                      # For feasibility debugging, set to true to return model without solving
        gurobiInstance = 0,                     # If running multiple solves in sequence, pass in Gurobi instance for slight speed up
        printout = :off                         # :on, :off
    )


    # number of points etc
    # --------------------

    I = -1:N

    dim_G = N+2

    𝐱, 𝐛, _, _ = data_generator_EpiSmooth(N, h, β; momentum=useMomentum)

    # define the model
    # ----------------
    if gurobiInstance == 0
        gurobiInstance = Gurobi.Env()
    end
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer( gurobiInstance)))
    if printout == :off
        set_silent(model)
    end

    # Gurobi settings
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", 1e-7)
    set_optimizer_attribute(model, "FeasibilityTol", 1e-7)
    set_optimizer_attribute(model, "OptimalityTol", 1e-7)


    # add the variables
    # -----------------

    # construct G ⪰ 0
    @variable(model, G[1:dim_G, 1:dim_G], Symmetric)

    @variable(model, H[1:dim_G, 1:dim_G])

    @variable(model, F[-1:N])
    @variable(model, T[-1:N])

    # Set f_⋆ = 0
    fix(F[-1], 0)

    
    #Set objective
    @objective(model, Max, F[N]) #Max f_N

    # Epigraphically smooth constraints
    setEpiSmoothConstraints(model, I, β, G, F, T, 𝐱, 𝐛)

    # Normalization constraint
    for i in I
        @constraint(model, tr(G*(⊙(𝐛[:,i], 𝐛[:,i]))) + T[i]*T[i] == 1)
    end

    # Initial distance constraint
    @constraint(model, tr(G*B_mat(-1,0,h,𝐱)) <= R^2)

    # t_⋆ constraint
    @constraint(model, T[-1] == -1)

    # t_i negativity constraint
    for i=-1:N
        @constraint(model, T[i] <= 0)
    end

    # Enforce G = H^T H 
    for i in 1:dim_G
        for j in 1:dim_G
            if i < j
                fix(H[i,j], 0; force = true)
            end
        end
    end

    # diagonal components of L_cholesky are non-negative
    for i in 1:dim_G
        @constraint(model, H[i,i] >= 0)
    end

    @constraint(model, vectorize(G - (H * H'), SymmetricMatrixShape(dim_G)) .== 0)

    # Optional: Sanity check to catch cases where h is not sufficiently regular and problem is unbounded
    for i=0:N
        @constraint(model, F[i] <= 100)
    end


    # optimize
    # ----------------

    if modelOnly
        return -1, -1, -1, -1, -1, -1, model, 𝐱, 𝐛
    end

    optimize!(model)

    # store and return the solution
    # -----------------------------
    #print(primal_status(model))

    discard = primal_status(model) != MOI.FEASIBLE_POINT

    if discard
        p_star = 100
        G_star = -1*ones(size(G))
        F_star = -1*ones(size(F))
        T_star = -1*ones(size(T))
        H_star = -1*ones(size(H))
    else
        p_star = objective_value(model)
        G_star = value.(G)
        F_star = value.(F)
        T_star = value.(T)
        H_star = value.(H)
    end

    return p_star, G_star, F_star, T_star, H_star, discard, model, 𝐱, 𝐛

end


# Apply epismooth function constraints
function setEpiSmoothConstraints(model, I, β, G, F, T, 𝐱, 𝐛)
    for i in I
        for j in I
            @constraint(model,
                tr(G*(⊙(𝐛[:,i], 𝐱[:,j] - 1/β*𝐛[:,j] - 𝐱[:,i] + 1/β*𝐛[:,i]))) +
                F[j]*T[i] - 1/β*T[i]*T[j] - F[i]*T[i] + 1/β*T[i]*T[i]
                <= 0
                )
        end
    end
end

# Generate weights for Nesterov momentum
function generateMomentumWeights_Nesterov(N)
    zeta = zeros(N+1)
    zeta = OffsetArray(zeta,0:N)

    zeta[0] = 1
    for i=1:N
        zeta[i] = (1+sqrt(1+4*zeta[i-1]^2))/2
    end

    return zeta
end