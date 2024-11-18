#Data Collection

include("FW_Main.jl")
include("FW_GlobalSolve.jl")

using Optim


# Solve a sequence of FW PEPs using local solve method.
# - param: Determines the parameter you are collecting data across (:N, :β, :α, :L, :μ)
# - paramVals: array of values to test for parameter
# All other parameters are fixed at the input values while param varies.

function collectData(param, paramVals, N, D, L, μ, α, β, δ, 
        ℓ,                                                      # Determines stepsizes: h_k = ℓ/(k+ℓ)
        functionType,                                           # :smooth, :smoothSC
        setType,                                                # :convex, :smooth, :SC, :smoothSC
        optLoc,                                                 # :exterior, :inSet, :interior
        SCDiam                                                  # :large, :small - Determines choice of λ for diameter constraints. :large corresponds to λ=1 and results in an upper bound for the true PEP result. :small corresponds to λ=1/sqrt(2) and results in a lower bound for the true PEP result.
    )     

    dataVals = zeros(length(paramVals))
    isFinalIterateBestData = zeros(length(paramVals))

    global initPt

    # Initial guess at ||g_i|| values for optimal solution. One can try different heuristics.
    if param==:N
        initPt = 0.2*ones(minimum(paramVals)+1)
        dataSqVals = zeros(length(paramVals), maximum(paramVals)+2)
    else
        initPt = 0.2*ones(N+1)
        dataSqVals = zeros(length(paramVals), N+2)
    end

    # For :inSet, :interior we exclude ||g_⋆|| since it is zero
    if optLoc in [:inSet, :interior]
        initPt = initPt[1:end-1]
    end

    for (j,paramVal) in enumerate(paramVals)
        global initPt

        if param == :N
            N = paramVal
        elseif param == :L
            L = paramVal
        elseif param == :μ
            μ = paramVal
        elseif param == :α
            α = paramVal
        elseif param == :β
            β = paramVal
        else
            @warn "Invalid parameter type"
            return 0
        end

        print(paramVal," ")
 
        objectiveType = :minIterate     # only :minIterate is supported for FW

        h = zeros(N)
        h = OffsetArray(h,0:N-1)

        if (β < α)&&(setType in [:smooth, :smoothSC])
            @warn "β must be larger than α"
        end
        if (2/β > D)&&(setType in [:smooth, :smoothSC])
            @warn "D must be larger than 2/β"
        end
        if (D > 2/α) && (SCDiam==:small)
            @warn "D must be smaller than 2/α"
        end
        if (sqrt(2)*D > 2/α) && (SCDiam == :large)
            @warn "If using large diam, then must have sqrt(2)*D < 2/α"
        end

    
        # Standard step sequence
        for i=0:N-1
            h[i] = ℓ/(i+ℓ)
        end

        sol, gSqAll, isFinalIterateBest = solveFWLocal(N, D, L, μ, α, β, δ, h, functionType, setType, objectiveType, optLoc, SCDiam; p0 = initPt)
    
        if param == :N
            initPt = gSqAll       # If we are looping on N, then use gSqN to form next guess
        else
            initPt = gSqAll[1:end-1]       # Otherwise, just use pOpt as our guess for the next param value
        end

        # Save off ||g_i||^2 data
        if optLoc in [:inSet, :interior]
            dataSqVals[j, 2:N+2] = gSqAll
        else
            dataSqVals[j, 1:N+2] = gSqAll
        end
    

        dataVals[j] = sol

        # Flag if final iterate was the unique minimum iterate
        if isFinalIterateBest 
            isFinalIterateBestData[j] = 1.0
        end

    end

    return dataVals, dataSqVals, isFinalIterateBestData
end

# Solve FW PEP using local solve method.
function solveFWLocal(N, D, L, μ, α, β, δ, h, functionType, setType, objectiveType, optLoc, SCDiam; p0 = 0)

    # If simple case, we can solve directly as an SDP
    if (setType == :convex)&&(optLoc in [:exterior, :inSet])
        sol, GOpt, _, discard, model_primal, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟, isFinalIterateBest = solve_primal_FW_Convex(N, h, D, L, μ; functionType = functionType, optLoc = optLoc, objectiveType = objectiveType)

        if optLoc == :inSet
            gSqAll = -1*ones(N+1) #Placeholder since this is not needed for the simple case
        else
            gSqAll = -1*ones(N+2) #Placeholder since this is not needed for the simple case
        end
    else
        # Otherwise, we optimize along the <g_i,n_i> = ||g_i|| curve

        if p0 == 0 # If we did not pass in initial guess p0, then update to default values
            if optLoc in [:inSet, :interior]
                p0 = 0.2*ones(N)
            else
                p0 = 0.2*ones(N+1)
            end
        end

        # Set objective for optimizer
        f(p) = -solve_primal_FW_OnCurve_VecWrapper(N, h, D, L, μ, α, β, δ, p; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = false, SCDiam = SCDiam)[1]

        # Upper/Lower bounds
        if optLoc in [:inSet, :interior]
            lower = zeros(N)
            upper = Inf*ones(N)
        else
            lower = zeros(N+1)
            upper = Inf*ones(N+1)
        end

        inner_optimizer = NelderMead()

        # Optimize
        result = Optim.optimize(f, lower, upper, p0, inner_optimizer, Optim.Options(show_trace = false, g_tol = 1e-5))

        pOpt = Optim.minimizer(result)


        # Solve objective using optimal solution in order to get GOpt
        _, GOpt, _, discard, model_primal, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟, isFinalIterateBest = solve_primal_FW_OnCurve_VecWrapper(N, h, D, L, μ, α, β, δ, pOpt; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = false, SCDiam = SCDiam)

        sol = -Optim.minimum(result)

        # Get ||g_N||^2
        gSqN = tr(GOpt*⊙(𝐠[:,N], 𝐠[:,N]))

        gSqAll = vcat(pOpt, gSqN)

    end

    return sol, gSqAll, isFinalIterateBest
end


# Solve a sequence of FW PEPs using global solve method.
# - param: Determines the parameter you are collecting data across (:N, :β, :α, :L, :μ)
# - paramVals: array of values to test for parameter
# All other parameters are fixed at the input values while param varies.
function collectData_Global(param, paramVals, N, D, L, μ, α, β, δ, 
        ℓ,                                                              # Determines stepsizes: h_k = ℓ/(k+ℓ)
        functionType,                                                   # :smooth, :smoothSC
        setType,                                                        # :convex, :smooth, :SC, :smoothSC
        optLoc,                                                         # :exterior, :inSet, :interior
        SCDiam;                                                         # :large, :small - Determines choice of λ for diameter constraints. :large corresponds to λ=1 and results in an upper bound for the true PEP result. :small corresponds to λ=1/sqrt(2) and results in a lower bound for the true PEP result.
        Δ = 1e-4,                                                       # Optimality precision
        lowerBoundMode = :localSolve,                                   # :localSolve, :value, :onTheFly
        lowerBoundVal = 0                                               # if lowerBoundMode == :value, this is used for lower bound
    )

    dataVals = zeros(length(paramVals))
    isFinalIterateBestData = zeros(length(paramVals))


    for (j,paramVal) in enumerate(paramVals)
        

        if param == :N
            N = paramVal
        elseif param == :L
            L = paramVal
        elseif param == :μ
            μ = paramVal
        elseif param == :α
            α = paramVal
        elseif param == :β
            β = paramVal
        else
            @warn "Invalid parameter type"
            return 0
        end

        print(paramVal," ")
 
        objectiveType = :minIterate

        h = zeros(N)
        h = OffsetArray(h,0:N-1)

        if (β < α)&&(setType in [:smooth, :smoothSC])
            @warn "β must be larger than α"
        end
        if (2/β > D)&&(setType in [:smooth, :smoothSC])
            @warn "D must be larger than 2/β"
        end
        if (D > 2/α) && (SCDiam==:small)
            @warn "D must be smaller than 2/α"
        end
        if (sqrt(2)*D > 2/α) && (SCDiam == :large)
            @warn "If using large diam, then must have sqrt(2)*D < 2/α"
        end
    
        # Standard step sequence
        for i=0:N-1
            h[i] = ℓ/(i+ℓ)
        end


        gMax = 1    # Determines initial branching location for g_i (heuristic to improve computation speed, does not affect output)

        # If simple case, we can solve directly as an SDP
        if (setType == :convex)&&(optLoc in [:exterior, :inSet])
            resultOpt, GOpt, _, discard, model_primal, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟, isFinalIterateBest = solve_primal_FW_Convex(N, h, D, L, μ; functionType = functionType, optLoc = optLoc, objectiveType = objectiveType)
        else
            # Run local solve to get lower bound
            if lowerBoundMode == :localSolve
                f(p) = -solve_primal_FW_OnCurve_VecWrapper(N, h, D, L, μ, α, β, δ, p; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, modelOnly = false, SCDiam = SCDiam)[1]

                if optLoc in [:inSet, :interior]
                    lower = zeros(N)
                    upper = Inf*ones(N)
                else
                    lower = zeros(N+1)
                    upper = Inf*ones(N+1)
                end

                if optLoc in [:inSet, :interior]
                    p0 = 0.2*ones(N)
                else
                    p0 = 0.2*ones(N+1)
                end
                inner_optimizer = NelderMead()

                solution = Optim.optimize(f, lower, upper, p0, inner_optimizer, Optim.Options(show_trace = false, g_tol = 1e-5))
                lowerBound_loc = -Optim.minimum(solution) #Make it positive again


                resultOpt, GOpt, _, _, _, _, _, _, iter, _, _, isFinalIterateBest = runGlobalOpt_FW(N, h, D, L, μ, α, β, δ, functionType, setType, :on; lowerBoundVal = lowerBound_loc, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam, gMax = gMax, maxIter = 1e6, Δ = Δ)
            
            elseif lowerBoundMode == :value
                # Use predetermined lower bound
                resultOpt, GOpt, _, _, _, _, _, _, iter, _, _, isFinalIterateBest = runGlobalOpt_FW(N, h, D, L, μ, α, β, δ, functionType, setType, :on; lowerBoundVal = lowerBoundVal, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam, gMax = gMax, maxIter = 1e6, Δ = Δ)
            else
                # lower bound calculated on-the-fly
                resultOpt, GOpt, _, _, _, _, _, _, iter, _, _, isFinalIterateBest = runGlobalOpt_FW(N, h, D, L, μ, α, β, δ, functionType, setType, :off; objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam, gMax = gMax, maxIter = 1e6, Δ = Δ)
            end

        end


        dataVals[j] = resultOpt

        if isFinalIterateBest==-1
            isFinalIterateBestData[j] = -1
        elseif isFinalIterateBest 
            isFinalIterateBestData[j] = 1.0
        end


    end

    return dataVals, isFinalIterateBestData
end
