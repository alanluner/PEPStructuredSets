include("SH_Main.jl")

betaVals = [2,3,5,10]

#--Trend with δ--
deltaVals = 0.15:0.05:0.5
deltaVals = deltaVals .+ 0.0001 #Shift values slightly so that things don't divide evenly - prevents rounding errors with N
data = zeros(length(betaVals), length(deltaVals))
for (i,β) in enumerate(betaVals)
    print(i)
    R = 1
    α = 0
    
    setType = :smooth

    for (j,δ) in enumerate(deltaVals)
        h = max(δ,1/β)
        NMaxGuess = Int(floor((R+h-δ)^2/h^2)) # Apply guess based on Theorem 6
        data[i,j] = solve_SH_HaltingProblem_WithGuess(R, α, β, δ, setType, NMaxGuess)
    end
end



