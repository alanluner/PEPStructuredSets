include("../../FrankWolfe/FW_Main.jl")
include("../../FrankWolfe/FW_DataCollection.jl")
include("FW_SmoothStepsizeImprovement.jl")
include("FW_ConvexStepsizeOptimization.jl")

# Settings
optLoc = :exterior                  # :exterior, :inSet
objectiveType = :finalIterate       # :finalIterate
functionType = :smooth              # :smooth (only smooth is supported)
setType = :smooth                   # :convex, :smooth
stepMode = :standard                # :standard, :matrix


# Parameter values
N = 5
D = 1
L = 1
β = 10


if stepMode == :matrix
    hOpt = OffsetArray(zeros(N,N),0:N-1, 0:N-1)
else
    hOpt = OffsetArray(zeros(N),0:N-1)
end

if setType == :smooth
    resultOpt, hOpt = solveOptimalStepSizeSmoothRelaxationFW(N, D, L, β; functionType = functionType, optLoc = optLoc, objectiveType = objectiveType, stepMode = stepMode)
    hOpt = OffsetArrays.no_offset_view(hOpt)
elseif setType == :convex
    resultOpt, hOpt = solveOptimalStepSizeConvexFW(N, D, L; functionType = functionType, optLoc = optLoc, objectiveType = objectiveType, stepMode = stepMode)
    hOpt = OffsetArrays.no_offset_view(hOpt)
end

print(resultOpt)

print("\n\n", hOpt)