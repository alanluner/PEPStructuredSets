include("AP_Main.jl")


# Solve global PEP for Alternating Projections
function runGlobalOpt_AP(N, R, α, δ, 
        setType,                            # :convex, :SC
        useLowerBound;                      # :on, :off
        lowerBoundVal = 0,                  # If useLowerBound=:on, algorithm will terminate when currVal < lowerBoundVal + Δ. Otherwise, lower bound is calculated on-the-fly
        uvStarMax = 10,                     # Determines initial branching location for u_⋆ and v_⋆ (heuristic to improve computation speed, does not affect output)
        uvMax = 1,                          # Determines initial branching location for u_i and v_i (heuristic to improve computation speed, does not affect output)
        maxIter = 1e6,                      # Maximum iterations
        splitHeuristic = 1,                 # Heuristic for splitting partition: 0 - Standard bisection, 1- Standard bisection with special handling near zero
        Δ = 1e-4                            # Optimality precision
    )

    global ptCounter = 1

    #---Initial SDP run---#
    (branchUInit, branchVInit) = getInitialBranch_AP(uvStarMax,uvMax,N)
    resultInit, GInit, _, _, _, _, 𝐮, 𝐯, 𝐦, 𝐧 = solve_primal_AP_WithCut(N, R, α, δ, setType, branchUInit, branchVInit)
    feasDistInit, flagInitUV, feasIndexInit, uSqInit, vSqInit = getFeasibilityDistance_AP(GInit, N, 𝐮, 𝐯, 𝐦, 𝐧)
    #print("Result Init: ", resultInit)

    #---Generate and store in dictionaries---#
    treeDataU = Dict(ptCounter => branchUInit)
    treeDataV = Dict(ptCounter => branchVInit)
    valueData = Dict(ptCounter => resultInit)
    feasData = Dict(ptCounter => [feasDistInit, string(feasIndexInit)*"_"*string(flagInitUV)])
    uSqData = Dict(ptCounter => uSqInit)
    vSqData = Dict(ptCounter => vSqInit)


    global iter = 0
    global ptID = ptCounter
    global idx = feasIndexInit
    global flagUV = flagInitUV
    global success = 0

    # Run through the remaining initial cuts (This is going to be 2^(2N+2) SDP solves)
    for num=1:2^(2N+2)-1
        global iter = 0
        global ptCounter

        # Convert to binary to encode the cut information (left [0] vs right [1] for each of 2N+2 values)
        binStr = bitstring(num)[end-(2N+1):end]
        binStrU = binStr[1:N+1]
        binStrV = binStr[N+2:end]

        binVecU = OffsetArray(parse.(Int, split(binStrU,"")), -1:N-1)

        #Special indexing (skip 0) for V array
        binVecV = OffsetArray(zeros(N+2),-1:N)
        binVecV[-1] = parse.(Int, split(binStrV,""))[1]
        binVecV[1:N] = parse.(Int, split(binStrV,""))[2:end]

        # Construct U branch based on binary encoding
        branchU = OffsetArray(zeros(N+1,2), -1:N-1, 1:2)
        for i=-1:N-1
            if binVecU[i] == 0
                if i==-1
                    branchU[i,1] = 0
                    branchU[i,2] = uvStarMax
                else
                    branchU[i,1] = 0
                    branchU[i,2] = uvMax
                end
            else
                if i==-1
                    branchU[i,1] = uvStarMax
                    branchU[i,2] = -1 #Flag that this is terminal cut
                else
                    branchU[i,1] = uvMax
                    branchU[i,2] = -1 #Flag that this is terminal cut
                end
            end
        end

        # Construct V branch based on binary encoding
        branchV = OffsetArray(zeros(N+2,2), -1:N, 1:2)
        for k in vcat(-1, 1:N)
            if binVecV[k] == 0
                if k==-1
                    branchV[k,1] = 0
                    branchV[k,2] = uvStarMax
                else
                    branchV[k,1] = 0
                    branchV[k,2] = uvMax
                end
            else
                if k==-1
                    branchV[k,1] = uvStarMax
                    branchV[k,2] = -1 #Flag that this is terminal cut
                else
                    branchV[k,1] = uvMax
                    branchV[k,2] = -1 #Flag that this is terminal cut
                end
            end
        end

        # Solve primal for each branch
        result, G, _, _, _, _, 𝐮, 𝐯, 𝐦, 𝐧 = solve_primal_AP_WithCut(N, R, α, δ, setType, branchU, branchV)
        feasDist, flagUV, feasIndex, uSq, vSq = getFeasibilityDistance_AP(G, N, 𝐮, 𝐯, 𝐦, 𝐧)

        iter = iter+1
        ptCounter = ptCounter+1

        # Add result and metadata to tree
        updateTree_AP(ptCounter, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, branchU, branchV, result, feasDist, feasIndex, flagUV, uSq, vSq)

    end

    # Find branch that achieves maximum value and save metadata
    ptID = getMaxPoint(valueData)
    idxString = feasData[ptID][2]
    idx = parse(Int, idxString[1:end-2])

    # flagUV determines whether the new splitting branch is in the U variable or the V variable
    if idxString[end] == 'u'
        flagUV = :u
    else
        flagUV = :v
    end

    global gap = -1 #Flag that gap has not yet been calculated

    # Now loop until we achieve desired tolerance (or maximum iterations)
    while iter < maxIter
        global iter += 1
        global ptCounter, idx, ptID, newPtID, success, gap


        if mod(iter, 250)==0
            print(iter, " ", valueData[ptID])
            print('\n')
            print("Gap: ", gap)
            print('\n')
        end

        
        branchDataU = getBranchData(treeDataU, ptID)
        branchDataV = getBranchData(treeDataV, ptID)

        #data = [a_-1 | a_0 | ... | a_N-1 
        #       b_-1 | b_0 | ... | b_N-1 ] [OffsetArray]

        if flagUV == :u
            branchData = copy(branchDataU)
        else
            branchData = copy(branchDataV)
        end

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

        if flagUV == :u
            resultLeft, G, discard = solve_primal_AP_WithCut(N, R, α, δ, setType, branchLeft, branchDataV)
        else
            resultLeft, G, discard = solve_primal_AP_WithCut(N, R, α, δ, setType, branchDataU, branchLeft)
        end

        if discard
            resultLeft = -1
            #print(branchLeft)
        end
        feasDistLeft, flagUVLeft, feasIndexLeft, uSqLeft, vSqLeft = getFeasibilityDistance_AP(G, N, 𝐮, 𝐯, 𝐦, 𝐧)


        if flagUV == :u
            resultRight, G, discard = solve_primal_AP_WithCut(N, R, α, δ, setType, branchRight, branchDataV)
        else
            resultRight, G, discard = solve_primal_AP_WithCut(N, R, α, δ, setType, branchDataU, branchRight)
        end


        if discard
            resultRight = -1.0
            #print(branchRight)
        end
        feasDistRight, flagUVRight, feasIndexRight, uSqRight, vSqRight = getFeasibilityDistance_AP(G, N, 𝐮, 𝐯, 𝐦, 𝐧)

        #---Update dictionaries for new branches---#
        ptCounter = ptCounter+1 #Increment for new ptID
        if flagUV == :u
            updateTree_AP(ptCounter, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, branchLeft, branchDataV, resultLeft, feasDistLeft, feasIndexLeft, flagUVLeft, uSqLeft, vSqLeft)
        else
            updateTree_AP(ptCounter, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, branchDataU, branchLeft, resultLeft, feasDistLeft, feasIndexLeft, flagUVLeft, uSqLeft, vSqLeft)
        end

        ptCounter = ptCounter+1 #Increment for new ptID
        if flagUV == :u
            updateTree_AP(ptCounter, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, branchRight, branchDataV, resultRight, feasDistRight, feasIndexRight, flagUVRight, uSqRight, vSqRight)
        else
            updateTree_AP(ptCounter, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, branchDataU, branchRight, resultRight, feasDistRight, feasIndexRight, flagUVRight, uSqRight, vSqRight)
        end

        #---Remove old point from dictionary, since it is no longer relevant---#
        removeFromTree_AP(ptID, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData)

        #---Find maximum value---#
        newPtID = getMaxPoint(valueData)

        #---Check if current max value is within desired tolerance of lower bound---#

        currVal = valueData[newPtID]

        if useLowerBound==:on
            lowerBound = lowerBoundVal
        else
            # If useLowerBound == :off, generate a lower bound on the fly by solving primal OnCurve with the current ||u|| and ||v|| values
            targetU = uSqData[newPtID]
            targetV = vSqData[newPtID]

            lowerBound, _, discard, _ = solve_primal_AP_OnCurve(N, R, α, δ, setType, targetU, targetV)
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
            # Update our new point of focus
            ptID = newPtID
            #Pull one of the infeasible dimensions to refine
            idxString = feasData[newPtID][2]
            # print(idxString[1:end-2])
            idx = parse(Int, idxString[1:end-2])
            if idxString[end] == 'u'
                flagUV = :u
            else
                flagUV = :v
            end
        end

    end

    # If successful, re-run the SDP for our worst-case point (so that we don't have to store the result for all of our runs)
    if success == 1 
        print("SuccessType: ",success,'\n')
        branchOptU = treeDataU[newPtID]
        branchOptV = treeDataV[newPtID]

        resultOpt, GOpt, discard = solve_primal_AP_WithCut(N, R, α, δ, setType, branchOptU, branchOptV)
        feasOpt = getFeasibilityDistance_AP(GOpt, N, 𝐮, 𝐯, 𝐦, 𝐧)
        cutFeas = getCutFeasibility_AP(N, branchOptU, branchOptV)
    else
        branchOptU = treeDataU[newPtID]
        branchOptV = treeDataV[newPtID]

        resultOpt = -1
        _, GOpt, _ = solve_primal_AP_WithCut(N, R, α, δ, setType, branchOptU, branchOptV)
        feasOpt = -1
        cutFeas = -1
        @warn "Unsuccessful, reached maximum iterations: " maxIter
    end

    return resultOpt, GOpt, feasOpt, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, iter, cutFeas

end


# Save off branch and value data into dictionaries
function updateTree_AP(ptID, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData, branchU, branchV, val, feasDist, feasIndex, flagUV, uSqVals, vSqVals)
    treeDataU[ptID] = branchU
    treeDataV[ptID] = branchV
    valueData[ptID] = val
    feasData[ptID] = [feasDist, string(feasIndex)*"_"*string(flagUV)]
    uSqData[ptID] = uSqVals
    vSqData[ptID] = vSqVals
end

# Remove a branch from dictionaries
function removeFromTree_AP(ptID, treeDataU, treeDataV, valueData, feasData, uSqData, vSqData)
    delete!(treeDataU, ptID)
    delete!(treeDataV, ptID)
    delete!(valueData, ptID)
    delete!(feasData, ptID)
    delete!(uSqData, ptID)
    delete!(vSqData, ptID)
end

function getBranchData(treeData, ptID)
    return treeData[ptID]
end

# Find maximum value over all current branches
function getMaxPoint(valueData)
    _, newPtID = findmax(valueData)
    return newPtID
end

# Initialize branches using our chosen upper bound on ||u|| and ||v||
function getInitialBranch_AP(uvStarMax,uvMax,N)
 
    branchU = zeros(N+1,2)
    branchU = OffsetArray(branchU, -1:N-1, 1:2)
    branchU[:,2] .= uvMax
    branchU[-1,2] = uvStarMax

    branchV = zeros(N+2,2)
    branchV = OffsetArray(branchV, -1:N, 1:2)
    branchV[:,2] .= uvMax
    branchV[-1,2] = uvStarMax
    branchV[0,2] = 0 #Ignore v_0

    return branchU, branchV
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

# Calculate distance from the <u_i, m_i> = ||u_i|| and <v_k, n_k> = ||v_k|| constraint curves
# Used to determine which index to split
function getFeasibilityDistance_AP(G, N, 𝐮, 𝐯, 𝐦, 𝐧)
    maxDist = 0
    idxU = -2
    idxV = -2
    uSqVals = OffsetArray(zeros(N+1), -1:N-1)
    vSqVals = OffsetArray(zeros(N+2), -1:N) #Make this one larger to account for v0

    I = -1:N-1
    for i in I  #Alternatively use collect(-1:N-1)[randperm(N+1)]
        MAT1 = ⊙(𝐮[:,i], 𝐮[:,i])
        MAT2 = ⊙(𝐮[:,i], 𝐦[:,i])

        #dist = ||u_i|| - ⟨u_i, m_i⟩
        dist = 0
        try
            dist = sqrt(tr(G*MAT1)) - tr(G*MAT2)
        catch e
            dist = 0 - tr(G*MAT2)
        end

        if dist > maxDist
            maxDist = dist
            idxU = i
        end

        # Also calculate and return ||u_i||^2 values
        uSqVals[i] = tr(G*MAT1)
    end

    K = vcat(-1, 1:N)
    for k in K 
        MAT1 = ⊙(𝐯[:,k], 𝐯[:,k])
        MAT2 = ⊙(𝐯[:,k], 𝐧[:,k])

        #dist = ||v_k|| - ⟨v_k, n_k⟩
        dist = 0
        try
            dist = sqrt(tr(G*MAT1)) - tr(G*MAT2)
        catch
            dist = 0 - tr(G*MAT2)
        end
        
        if dist > maxDist
            maxDist = dist
            idxU = -2 #Reset to -2 so that we make update to v not u
            idxV = k
        end

        # Also calculate and return ||v_k||^2 values
        vSqVals[k] = tr(G*MAT1)
    end

    if idxU == -2
        flagUV = :v
        idx = idxV
    else
        flagUV = :u
        idx = idxU
    end
    
    return maxDist, flagUV, idx, uSqVals, vSqVals

end


# Approximate measure of how close the current solution is to the desired <u_i,m_i> = ||u_i|| and <v_k,n_k> = ||v_k|| constraint curves
# Not used for the algorithm, but helps give a sense of what cut precision was necessary for algorithm to terminate. 
function getCutFeasibility_AP(N, branchU, branchV)

    cutFeas = 100
    for i in -1:N-1
        a = branchU[i,1]
        b = branchU[i,2]
        if b==-1
            #Then our worst case lies in a terminal cut, so it has infinite cut infeasibility. That's fine though, cutFeas is just to get a sense of things
            return cutFeas = -1
        end
        dist = (sqrt(a)+sqrt(b))/4 - sqrt(a*b)/(sqrt(a)+sqrt(b))
        if dist < cutFeas
            cutFeas = dist
        end
    end

    K = vcat(-1, 1:N)
    for k in K
        a = branchV[k,1]
        b = branchV[k,2]
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
        
    

