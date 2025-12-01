include("FW_DataCollection.jl")


# Settings
SCDiam = :large             # :large, :small - Determines choice of λ for diameter constraints. :large corresponds to λ=1 and results in an upper bound for the true PEP result. :small corresponds to λ=1/sqrt(2) and results in a lower bound for the true PEP result.
optLoc = :exterior          # :exterior, :inSet, :interior
functionType = :smooth      # :smooth, :smoothSC
setType = :smooth           # :convex, :smooth, :SC, :smoothSC
lowerBoundMode = :localSolve        # :localSolve, :value, :onTheFly

# Parameter values
NMax = 3            # Not recommended to exceed 4 or 5
D = 1
L = 1
ℓ = 2
δ = 0.25
μ = 0.5
β = 5
α = 1
Δ = 1e-4


#Parameter for sweep
param = :N
paramVals = 1:NMax

# Run PEP for N = 1,...,NMax
data, isFinalIterateBestData = collectData_Global(param, paramVals, 0, D, L, μ, α, β, δ, ℓ, functionType, setType, optLoc, SCDiam; Δ = Δ, lowerBoundMode = lowerBoundMode)