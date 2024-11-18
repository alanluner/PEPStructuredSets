include("ES_Main.jl")

N = 3


global gurobi = Gurobi.Env()

# Determine range of eta values
etaVals = 1 ./ collect(2:1:7) #Start at eta<1 so that opt stepsizes doesn't fail

dataGD1 = zeros(length(etaVals))
dataGDOpt = zeros(length(etaVals))
dataMomentum = zeros(length(etaVals))

for (i,η) in enumerate(etaVals)
    print('\n',i)

    R = η
    β = 1


    # Run with GD h=1
    h = OffsetArray(ones(N), 0:N-1)
    p, _ = solvePrimalEpiSmooth(N, R, β, h; modelOnly = false, gurobiInstance=gurobi, useMomentum=false, printout=:off)
    dataGD1[i] = p/η^2  # Rescale result

    # Run with silver stepsizes
    h = [sqrt(2), 2, sqrt(2)]
    h = OffsetArray(h,0:N-1)

    p, _ = solvePrimalEpiSmooth(N, R, β, h; modelOnly = false, gurobiInstance=gurobi, useMomentum=false, printout=:off)
    dataGDOpt[i] = p/η^2  # Rescale result

    # Nesterov Momentum
    h = OffsetArray(ones(N), 0:N-1)
    p, _ = solvePrimalEpiSmooth(N, R, β, h; modelOnly = false, gurobiInstance=gurobi, useMomentum=true, printout=:off)
    dataMomentum[i] = p/η^2   # Rescale result

end