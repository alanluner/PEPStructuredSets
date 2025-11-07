include("FW_DataCollection.jl")

# Settings
SCDiam = :large             # :large, :small - Determines choice of λ for diameter constraints. :large corresponds to λ=1 and results in an upper bound for the true PEP result. :small corresponds to λ=1/sqrt(2) and results in a lower bound for the true PEP result.
optLoc = :exterior          # :exterior, :inSet, :interior
functionType = :smooth      # :smooth, :smoothSC
setType = :smooth           # :convex, :smooth, :SC, :smoothSC

# Parameter values
NMax = 5                   # Not recommended to exceed 15
D = 1
L = 1
ℓ = 2
δ = 0.25
μ = 0.5
β = 5
α = 1

N = NMax

#Parameter for sweep
param = :N
paramVals = 1:NMax

# Solve PEP for N = 1,...,NMax
data, _, isFinalIterateBestData = collectData(param, paramVals, 0, D, L, μ, α, β, δ, ℓ, functionType, setType, optLoc, SCDiam)