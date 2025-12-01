include("FW_DataCollection.jl")

# Settings
SCDiam = :large


# Parameter default values (Only used when applicable)
NMax = 5            # Not recommended to exceed 15
R = 1
L = 1
μ = 0.5
β = 5
α = 1.0
δ = 0.25
ℓ = 2

#Parameter for sweep
param = :N
paramVals = 1:NMax

# Run across all settings 
surveyData = Dict()
for optLoc in [:interior, :exterior]
    for functionType in [:smooth, :smoothSC]
        for setType in [:convex, :smooth, :SC, :smoothSC]

            data, _, isFinalIterateBestData = collectData(param, paramVals, 0, R, L, μ, α, β, δ, ℓ, functionType, setType, optLoc, SCDiam)

            str = "F_" * string(functionType) * "__C_" * string(setType) * "__Opt_" * string(optLoc)

            surveyData[str] = data

        end
    end
end