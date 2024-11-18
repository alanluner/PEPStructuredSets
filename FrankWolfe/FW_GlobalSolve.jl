include("FW_Main.jl")


function runGlobalOpt_FW(N, h, D, L, μ, α, β, δ, 
        functionType,                               # :smooth, :smoothSC 
        setType,                                    # :convex, :smooth, :SC, :smoothSC
        useLowerBound;                              # :on, :off
        lowerBoundVal = 0,                          # If useLowerBound=:on, algorithm will terminate when currVal < lowerBoundVal + Δ. Otherwise, lower bound is calculated on-the-fly
        objectiveType = :minIterate,                # :minIterate
        optLoc = :exterior,                         # :exterior, :inSet, :interior
        SCDiam = :large,                            # :large, :small - Determines choice of λ for diameter constraints. :large corresponds to λ=1 and results in an upper bound for the true PEP result. :small corresponds to λ=1/sqrt(2) and results in a lower bound for the true PEP result.
        gMax = 2,                                   # Determines initial branching location for g_i (heuristic to improve computation speed, does not affect output)
        maxIter = 1e6,                              # Maximum iterations
        splitHeuristic = 1,                         # Heuristic for splitting partition: 0 - Standard bisection, 1- Standard bisection with special handling near zero
        Δ = 1e-5                                    # Optimality precision
    )

    # If simple case, we can solve directly as an SDP
    if (setType==:convex)&&(optLoc in [:exterior, :inSet])
        resultOpt, GOpt, FOpt, _ =  solve_primal_FW_Convex(N, h, D, L, μ; functionType = functionType, optLoc = optLoc, objectiveType = objectiveType)

        return resultOpt, GOpt, FOpt, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    end

    global ptCounter = 1

    #---Initial SDP run---#
    branchInit = getInitialBranch(gMax, N)
    resultInit, GInit, _, _, _, _, 𝐠, _, 𝐧, _, _ = solve_primal_FW_WithCut(N, h, D, L, μ, α, β, δ, branchInit; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam)
    feasDistInit, feasIndexInit, gSqInit = getFeasibilityDistance(GInit, N, 𝐠, 𝐧, optLoc)

    #---Generate and store in dictionaries---#
    treeData = Dict(ptCounter => branchInit)
    valueData = Dict(ptCounter => resultInit)
    feasData = Dict(ptCounter => [feasDistInit, feasIndexInit])
    gSqData = Dict(ptCounter => gSqInit)

    global iter = 0
    global ptID = ptCounter
    global idx = feasIndexInit
    global success = 0


    # Run through the remaining initial cuts (This is going to be 2^(N+1) SDP solves)
    global testMax = 0
    for num=1:2^(N+1)-1
        global iter = 0
        global ptCounter

        # Convert to binary to encode each cut option (left [0] vs right[1] for all N+1 indices)
        binStr = bitstring(num)[end-N:end]
        binVec = parse.(Int, split(binStr,""))
        binVec = OffsetArray(binVec, -1:N-1)

        # Construct branch based on binary encoding
        branch = OffsetArray(zeros(N+1,2), -1:N-1, 1:2)
        for i=-1:N-1
            if binVec[i] == 0
                branch[i,1] = 0
                branch[i,2] = gMax
            else
                branch[i,1] = gMax
                branch[i,2] = -1 #Flag that this is terminal cut
            end
        end

        # Solve primal for the branch
        result, G, _, _, _, _, 𝐠, _, 𝐧, _, _ = solve_primal_FW_WithCut(N, h, D, L, μ, α, β, δ, branchInit; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam)
        feasDist, feasIndex, gSq = getFeasibilityDistance(G, N, 𝐠, 𝐧, optLoc)

        iter = iter+1
        ptCounter = ptCounter+1

        # Add result and metadata to tree
        updateTree(ptCounter, treeData, valueData, feasData, gSqData, branch, result, feasDist, feasIndex, gSq)

    end

    # Find branch that achieves maximum value and save metadata
    ptID = getMaxPoint(valueData)
    idx = Int(feasData[ptID][2])


    global gap = -1 # Flag that gap has not yet been calculated

    # Now loop until we achieve desired tolerance (or maximum iterations)
    while iter < maxIter
        global iter += 1
        global ptCounter, idx, ptID, newPtID, success
        global gap

        if mod(iter, 250)==0
            print(iter, " ", valueData[ptID])
            print('\n')
            print("Gap: ",gap)
            print('\n')
            print(gSqData[ptID])
            print('\n')
        end

        branchData = getBranchData(treeData, ptID)

        #data = [a_-1 | a_0 | ... | a_N-1 
        #       b_-1 | b_0 | ... | b_N-1 ] [OffsetArray]
        a = branchData[idx,1]
        b = branchData[idx,2]

        #---Split and determine new bounds---#

        a_left, b_left, a_right, b_right = getSplit(a,b; heuristic = splitHeuristic)

        #---Create new branches ---#
        branchLeft = copy(branchData) #Copy so we aren't just creating a new reference
        branchLeft[idx,1] = a_left 
        branchLeft[idx,2] = b_left

        branchRight = copy(branchData) #Copy so we aren't just creating a new reference
        branchRight[idx,1] = a_right
        branchRight[idx,2] = b_right


        #---Run SDP for left and right---#
        resultLeft, G, F, discard = solve_primal_FW_WithCut(N, h, D, L, μ, α, β, δ, branchLeft; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam)
        if discard
            resultLeft = -1
            #print(branchLeft)
        end
        feasDistLeft, feasIndexLeft, gSqLeft = getFeasibilityDistance(G, N, 𝐠, 𝐧, optLoc)

        resultRight, G, F, discard = solve_primal_FW_WithCut(N, h, D, L, μ, α, β, δ, branchRight; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam)
        if discard
            resultRight = -1.0
            #print(branchRight)
        end
        feasDistRight, feasIndexRight, gSqRight = getFeasibilityDistance(G, N, 𝐠, 𝐧, optLoc)

        #---Update dictionaries for new branches---#
        ptCounter = ptCounter+1 #Increment for new ptID
        updateTree(ptCounter, treeData, valueData, feasData, gSqData, branchLeft, resultLeft, feasDistLeft, feasIndexLeft, gSqLeft)
        ptCounter = ptCounter+1 #Increment for new ptID
        updateTree(ptCounter, treeData, valueData, feasData, gSqData, branchRight, resultRight, feasDistRight, feasIndexRight, gSqRight)

        #---Remove old point from dictionary, since it is no longer relevant---#
        removeFromTree(ptID, treeData, valueData, feasData, gSqData)

        #---Find maximum value---#
        newPtID = getMaxPoint(valueData)

        #---Check if current max value is within desired tolerance of lower bound---#

        currVal = valueData[newPtID]

        if useLowerBound == :on
            lowerBound = lowerBoundVal
        else
            # If useLowerBound == :off, generate a lower bound on the fly by solving primal OnCurve with the current ||g|| values
            target = gSqData[newPtID]
            lowerBound, _, _, discard = solve_primal_FW_OnCurve(N, h, D, L, μ, α, β, δ, target; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam)
        end

        # Calculate gap and check if below desired tolerance
        gap = currVal - lowerBound
        
        if currVal <= lowerBound + Δ
            # print("CurrVal:", currVal,'\n')
            # print("Lower Bound:",lowerBound,'\n')
            # print("Delta",Δ,'\n')
            success = 1
            break
        else
            success == 0
            # Update our new point of focus
            ptID = newPtID
            #Pull one of the infeasible dimensions to refine
            idx = Int(feasData[newPtID][2])
        end

    end

    currVal = valueData[newPtID]

    # If successful, re-run the SDP for our worst-case point (so that we don't have to store the result for all of our runs)
    if success == 1 
        branchOpt = treeData[newPtID]
        resultOpt, GOpt, FOpt, discard = solve_primal_FW_WithCut(N, h, D, L, μ, α, β, δ, branchOpt; functionType = functionType, setType = setType, objectiveType = objectiveType, optLoc = optLoc, SCDiam = SCDiam)
        feasOpt = getFeasibilityDistance(GOpt, N, 𝐠, 𝐧, optLoc)
        cutFeas = getCutFeasibility(branchOpt, optLoc, N)
        upperBd = -1

        # We want to check if final iterate is the minimum
        isFinalIterateBest = minimum(FOpt[1:N]) >= (1.01)*FOpt[N+1]
    else
        resultOpt = currVal
        GOpt = -1
        FOpt = -1
        feasOpt = -1
        cutFeas = -1
        upperBd = -1
        @warn "Unsuccessful, reached maximum iterations: " maxIter

        isFinalIterateBest = -1
    end


    return resultOpt, GOpt, FOpt, feasOpt, treeData, valueData, feasData, gSqData, iter, cutFeas, upperBd, isFinalIterateBest

end



# Save off branch and value data into dictionaries
function updateTree(ptID, treeData, valueData, feasData, gSqData, branch, val, feasDist, feasIndex, gSqVals)
    treeData[ptID] = branch
    valueData[ptID] = val
    feasData[ptID] = [feasDist, feasIndex]
    gSqData[ptID] = gSqVals
end

# Remove a branch from dictionaries
function removeFromTree(ptID, treeData, valueData, feasData, gSqData)
    delete!(treeData, ptID)
    delete!(valueData, ptID)
    delete!(feasData, ptID)
    delete!(gSqData, ptID)
end

function getBranchData(treeData, ptID)
    return treeData[ptID]
end

# Find maximum value over all current branches
function getMaxPoint(valueData)
    _, newPtID = findmax(valueData)
    return newPtID
end


# Initialize branch using our chosen upper bound on ||g||
function getInitialBranch(gMax,N)
    branch = zeros(N+1,2)
    branch = OffsetArray(branch, -1:N-1, 1:2)
    branch[:,2] .= gMax

    return branch
end

# Determine splitting values for branch
function getSplit(a,b; heuristic = 1, ϵ = 1e-4)

    #Handle terminal horizontal cut first
    if b==-1
        a_left = a
        b_left = 2*a
        a_right = 2*a
        b_right = -1

        return a_left, b_left, a_right, b_right
    end

    #Otherwise, proceed to normal cut options

    #Standard bisection
    if heuristic == 0
        splitPoint = (a+b)/2

    #Standard bisection with special handling near zero
    elseif heuristic == 1
        specialBd = 16*ϵ^2
        if (a==0)&&(b>specialBd)
            splitPoint = specialBd
        else
            splitPoint = (a+b)/2
        end
    end

    a_left = a
    b_left = splitPoint
    a_right = splitPoint
    b_right = b

    return a_left, b_left, a_right, b_right

end

# Calculate distance from the <g_i, n_i> = ||g_i|| constraint curve
# Used to determine which index to split
function getFeasibilityDistance(G, N, 𝐠, 𝐧, optLoc)
    maxDist = 0
    
    if optLoc in [:inSet, :interior]
        idx = 0
        idxSet = 0:N-1
    else
        idx = -1
        idxSet = -1:N-1
    end

    gSqVals = OffsetArray(zeros(N+1), -1:N-1)
    for i in idxSet
        MAT1 = ⊙(𝐠[:,i], 𝐠[:,i])
        MAT2 = ⊙(-𝐠[:,i], 𝐧[:,i])

        #dist = ||g_i|| - ⟨-g_i, n_i⟩
        dist = sqrt(abs(tr(G*MAT1))) - tr(G*MAT2) #Use abs to handle numerical error around zero
        if dist > maxDist
            maxDist = dist
            idx = i
        end

        # Also calculate and return ||g_i||^2 values
        gSqVals[i] = tr(G*MAT1)
    end
    
    return maxDist, idx, gSqVals

end


        
# Approximate measure of how close the current solution is to the desired <g_i,n_i> = ||g_i|| constraint curve
# Not used for the algorithm, but helps give a sense of what cut precision was necessary for algorithm to terminate. 
function getCutFeasibility(branch, optLoc, N)
    if optLoc in [:inSet,:interior]
        idxSet = 0:N-1
    else
        idxSet = -1:N-1
    end
    cutFeas = 100
    for i in idxSet
        a = branch[i,1]
        b = branch[i,2]
        if b==-1
            #Then our worst case lies in a terminal cut, so it has infinite cut infeasibility. That's fine though, cutFeas is just to get a sense of things
            return cutFeas = -1
        end
        dist = (sqrt(a)+sqrt(b))/4 - sqrt(a*b)/(sqrt(a)+sqrt(b))
        if dist < cutFeas
            cutFeas = dist
        end
    end

    return cutFeas
end

    

