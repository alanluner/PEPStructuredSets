include("AP_Main.jl")
include("AP_GlobalSolve.jl")

# Parameter Values
NMax = 3            # Maximum N for global solve. Not recommended to exceed 5.
R = 1               
δ = 0.25 

# Settings
uvStarMax = 100     # Determines initial branching location for u_⋆ and v_⋆ (heuristic to improve computation speed, does not affect output)
uvMax = 10          # Determines initial branching location for u_i and v_i (heuristic to improve computation speed, does not affect output)
maxIter = 1e5       # Maximum iterations
setType = :convex   # :convex, :SC (strongly convex not recommended due to much slower computation)
Δ = 1e-4            # Tolerance for solution


NVals = 1:NMax
data = zeros(length(NVals))

for (j,N) = enumerate(NVals)

    print(N," ")

    AP_LowerBound, _ = getAPLowerBound(N, R, δ)
    resultOpt, GOpt, feasOpt, _, _, _, _, _, _, iteration, cutFeas = runGlobalOpt_AP(N, R, 0, δ, setType, :on; lowerBoundVal = AP_LowerBound,  uvStarMax = uvStarMax, uvMax = uvMax, maxIter = maxIter, Δ = Δ, splitHeuristic = 0)

    print(resultOpt,'\n')
    print(AP_LowerBound,'\n')

    data[j] = resultOpt

end

