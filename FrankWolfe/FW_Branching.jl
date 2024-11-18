###-----Functions for generating selection vectors for G-----###

function data_generator_SetConvex(N, h)

    dim_G = 2N+3
    dim_F = N+1

    
    # define all the bold vectors
    # --------------------------


    # define 𝐱_0 and 𝐱_star


    𝐱_0 = e_i(dim_G, 1)

    𝐱_star = zeros(dim_G, 1)

    # define 𝐠_0, 𝐠_1, …, 𝐠_N

    # first define the 𝐠 vectors,
    # index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

    # 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]
    𝐠 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in -1:N #Start at -1, since we no longer assume g_⋆=0
        𝐠[:,i] = e_i(dim_G, i+3) #Now we shift by 3 because we need to include g_⋆
    end

    # 𝐳 = [𝐳_⋆ 𝐳_0 𝐳_1 𝐳_2 ... 𝐳_N-1] #Include z_⋆ which is just equal to x_⋆ = 0
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in 0:N-1
        𝐳[:,i] = e_i(dim_G, i+(N+4)) #Shift by N+4 to account for the 𝐠 vectors
    end


    # define the 𝐟 vectors

    # 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_F, N+2), 1:dim_F, -1:N)

    for i in 0:N
        𝐟[:,i] = e_i(dim_F, i+1)
    end


    # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}]
    𝐱 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)

    # assign values using our formula for 𝐱_k
    𝐱[:,0] = 𝐱_0

    for k in 1:N
        𝐱[:,k] = 𝐱[:,k-1] + h[k-1]*(𝐳[:,k-1] - 𝐱[:,k-1])
    end


    𝐧 = -1
    𝐰 = -1

    return 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟

end


function data_generator_SetSmooth(N, h)

    dim_G = 4N+6

    dim_F = N+1

    
    # define all the bold vectors
    # --------------------------


    # define 𝐱_0 and 𝐱_star


    𝐱_0 = e_i(dim_G, 1)

    𝐱_star = zeros(dim_G, 1)

    # define 𝐠_⋆ 𝐠_0, 𝐠_1, …, 𝐠_N

    # first we define the 𝐠 vectors,
    # index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

    # 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]
    𝐠 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in -1:N
        𝐠[:,i] = e_i(dim_G, i+3)
    end

    # 𝐳 = [𝐳_0 𝐳_1 𝐳_2 ... 𝐳_N-1]
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in 0:N-1
        𝐳[:,i] = e_i(dim_G, i+(N+4)) #Shift by N+4 to account for the 𝐠 vectors
    end

    # 𝐧 = [𝐧_⋆ 𝐧_0 𝐧_1 𝐧_2 ... 𝐧_N-1]
    𝐧 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in -1:N-1
        𝐧[:,i] = e_i(dim_G, i+(2N+5)) #Shift by 2N+4(+1) to account for the 𝐠 and 𝐳 vectors
    end

    # 𝐰 = [𝐰_⋆ 𝐰_0 𝐰_1 𝐰_2 ... 𝐰_N]
    𝐰 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in -1:N
        𝐰[:,i] = e_i(dim_G, i+(3N+6)) #Shift by 3N+4 (+1) to account for the 𝐠 and 𝐳 vectors
    end


    # define the 𝐟 vectors

    # 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_F, N+2), 1:dim_F, -1:N)

    for i in 0:N
        𝐟[:,i] = e_i(dim_F, i+1)
    end

    # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}]
    𝐱 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)

    # assign values using our formula for 𝐱_k
    𝐱[:,0] = 𝐱_0

    for k in 1:N
        𝐱[:,k] = 𝐱[:,k-1] + h[k-1]*(𝐳[:,k-1] - 𝐱[:,k-1])
    end
    
    return 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟

end


function data_generator_SetSC(N, h)

    dim_G = 3N+4

    dim_F = N+1

    
    # define all the bold vectors
    # --------------------------


    # define 𝐱_0 and 𝐱_star


    𝐱_0 = e_i(dim_G, 1)

    𝐱_star = zeros(dim_G, 1)

    # define 𝐠_⋆, 𝐠_0, 𝐠_1, …, 𝐠_N

    # first we define the 𝐠 vectors,
    # index -1 corresponds to ⋆, i.e.,  𝐟[:,-1] =  𝐟_⋆ = 0

    # 𝐠= [𝐠_⋆ 𝐠_0 𝐠_1 𝐠_2 ... 𝐠_N]
    𝐠 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)
    for i in -1:N
        𝐠[:,i] = e_i(dim_G, i+3)
    end

    # 𝐳 = [𝐳_⋆, 𝐳_0 𝐳_1 𝐳_2 ... 𝐳_N-1]
    𝐳 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in 0:N-1
        𝐳[:,i] = e_i(dim_G, i+(N+4)) #Shift by N+4 to account for the 𝐠 vectors
    end

    # 𝐧 = [𝐧_⋆ 𝐧_0 𝐧_1 𝐧_2 ... 𝐧_N-1]
    𝐧 = OffsetArray(zeros(dim_G, N+1), 1:dim_G, -1:N-1)
    for i in -1:N-1
        𝐧[:,i] = e_i(dim_G, i+(2N+5)) #Shift by 2N+4(+1) to account for the 𝐠 and 𝐳 vectors
    end

    # time to define the 𝐟 vectors

    # 𝐟 = [𝐟_⋆ 𝐟_0, 𝐟_1, …, 𝐟_N]

    𝐟 = OffsetArray(zeros(dim_F, N+2), 1:dim_F, -1:N)

    for i in 0:N
        𝐟[:,i] = e_i(dim_F, i+1)
    end

    # 𝐱 = [𝐱_{-1}=𝐱_⋆ ∣ 𝐱_{0} ∣ 𝐱_{1} ∣ … 𝐱_{N}]
    𝐱 = OffsetArray(zeros(dim_G, N+2), 1:dim_G, -1:N)

    # assign values next using our formula for 𝐱_k
    𝐱[:,0] = 𝐱_0

    for k in 1:N
        𝐱[:,k] = 𝐱[:,k-1] + h[k-1]*(𝐳[:,k-1] - 𝐱[:,k-1])
    end

    𝐰 = -1
    
    return 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, 𝐟

end

function data_generator_SetSmoothSC(N, h)
    return data_generator_SetSmooth(N,h) #Identical to data structure for smooth
end


function data_generator_SetConvex_Interior(N, h)
    return data_generator_SetSC(N,h)
end


###-----Functions for applying set interpolation constraints-----###

function applySetConstraints(model, setType, N, D, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, α, β, δ; optLoc = :exterior, SCDiam = :large)


    if optLoc == :inSet
        I = 0:N-1   #If inSet mode, then z_⋆ and n_⋆ are irrelevant, so we should not involve them in any of the standard constraints
        K = -1:N    #Use full constraints for x_⋆

        # Set gradient to zero
        if setType != :convex
            @constraint(model, tr(G*(⊙(𝐧[:,-1], 𝐧[:,-1]))) == 0)
        end
        @constraint(model, tr(G*(⊙(𝐠[:,-1], 𝐠[:,-1]))) == 0)
    
    elseif optLoc == :interior
        I = 0:N-1   #If interior mode, then z_⋆ and n_⋆ are irrelevant, so we should not involve them in any of the standard constraints
        K = 0:N     # Skip standard constraints for x_⋆, since we will use special interior constraints

        # Set gradient to zero
        @constraint(model, tr(G*(⊙(𝐧[:,-1], 𝐧[:,-1]))) == 0)     # For interior mode, we use 𝐧 even for convex
        @constraint(model, tr(G*(⊙(𝐠[:,-1], 𝐠[:,-1]))) == 0)

        # Apply interior constraints to x_⋆
        setConstraintsOptInterior(model, setType, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, α, β, δ)

    else
        I = -1:N-1
        K = -1:N
    end

    if setType == :convex
        setConstraints_Convex(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧)
    elseif setType == :smooth
        setConstraints_Smooth(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, β)
    elseif setType == :SC
        setConstraints_SC(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, α; SCDiam = SCDiam)
    elseif setType == :smoothSC
        setConstraints_SmoothSC(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, α, β; SCDiam = SCDiam)
    end

end

function setConstraints_Convex(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧)

    #I,I Constraints [Only involve z,g,n]
    for i in I
        for j in I
            if i==j
                continue
            end
            #[1] (Interp1)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐳[:,j]-𝐳[:,i]))) <= 0)

            #[4] (Interp4)
            @constraint(model, tr(G*(⊙(𝐳[:,i]-𝐳[:,j], 𝐳[:,i]-𝐳[:,j]))) <= D^2  )
        end
    end

    #I,K Constraints
    for i in I
        for k in K
            #[3] (Interp2)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐱[:,k]-𝐳[:,i]))) <= 0) 
            
            #[4] (Interp5)
            @constraint(model, tr(G*(⊙(𝐱[:,k]-𝐳[:,i], 𝐱[:,k]-𝐳[:,i]))) <= D^2  )
        end
    end

    #K,K Constraints
    for k in K
        for l in K
            if k==l
                continue
            end
            #[5] (Interp6)
            @constraint(model, tr(G*(⊙(𝐱[:,k]-𝐱[:,l], 𝐱[:,k]-𝐱[:,l]))) <= D^2  )
        end
    end

end

function setConstraints_Smooth(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, β)
    
    #I,I Constraints [Only involve g,z,n]
    for i in I
        for j in I
            if i==j
                continue
            end

            #[1] (Interp1)
            @constraint(model, tr(G*(⊙(-𝐠[:,i],  𝐳[:,j] - 1/β*𝐧[:,j] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= 0)

            #[2] (Interp4))
            @constraint(model, tr(G*(⊙(𝐳[:,i] - 1/β*𝐧[:,i] - 𝐳[:,j] + 1/β*𝐧[:,j],  𝐳[:,i] - 1/β*𝐧[:,i] - 𝐳[:,j] + 1/β*𝐧[:,j])))  <= (D-2/β)^2 )

            #[3] (Relaxation of n = -g/||g||)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐧[:,j] - 𝐧[:,i] ))) <= 0)
        end
    end

    # I constraints [Only involve g,z,n]
    for i in I
        #[4] (Relaxation of n = -g/||g||)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐧[:,i]))) == 1)

        #[5] (Relaxation of n = -g/||g||)
        @constraint(model, tr(G*(⊙(𝐠[:,i], 𝐧[:,i]))) <= 0 )
    end

    #I,K constraints
    for i in I
        for k in K
            #***INCLUDE case where i=k, it is necessary***

            #[6] (Interp2)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐰[:,k] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= 0 )

            #[7] (Interp5)
            @constraint(model, tr(G*(⊙(𝐰[:,k] - 𝐳[:,i] + 1/β*𝐧[:,i], 𝐰[:,k] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= (D-2/β)^2)
        end
    end

    #K,K constraints
    for k in K
        for l in K
            if k==l
                continue
            end
            #[8] (Interp6)
            @constraint(model, tr(G*(⊙(𝐰[:,k] - 𝐰[:,l], 𝐰[:,k] - 𝐰[:,l]))) <= (D-2/β)^2 )
        end
    end

    #K constraints
    for k in K
        #[9] (Interp3)
        @constraint(model, tr(G*(⊙(𝐱[:,k] - 𝐰[:,k], 𝐱[:,k] - 𝐰[:,k]))) <= 1/β^2)
    end

end

function setConstraints_SC(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, α; SCDiam = :large)

    if SCDiam == :large
        λ = 1
    elseif SCDiam == :small
        λ = 1/sqrt(2)
    end

    #I,I Constraints [Only involve z,g,n]
    for i in I
        for j in I
            if i==j
                continue
            end

            #[1] (Interp1)
            @constraint(model, tr(G*(⊙(𝐳[:,j]-𝐳[:,i]+1/α*𝐧[:,i],  𝐳[:,j]-𝐳[:,i]+1/α*𝐧[:,i]))) <= 1/α^2 )  # Strongly convex constraint
            
            #[2] (Relaxation of n = -g/||g||)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐧[:,j] - 𝐧[:,i] ))) <= 0)

            if SCDiam in [:small, :large]
                #[3] (Interp4)
                @constraint(model, tr(G*(⊙(𝐳[:,i] - 𝐳[:,j], 𝐳[:,i] - 𝐳[:,j]))) <= (λ*D)^2)
            end
        end
    end

    # I Constraints [Only involve z,g,n]
    for i in I
        #[4] (Relaxation of n = -g/||g||)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐧[:,i]))) == 1)

        #[5] (Relaxation of n = -g/||g||)
        @constraint(model, tr(G*(⊙(𝐠[:,i], 𝐧[:,i]))) <= 0 )
    end

    #I,K Constraints
    for i in I
        for k in K
            #***INCLUDE case where i=k, it is necessary***

            #[6] (Interp2)
            @constraint(model, tr(G*(⊙(𝐱[:,k]-𝐳[:,i]+1/α*𝐧[:,i], 𝐱[:,k]-𝐳[:,i]+1/α*𝐧[:,i]))) <= 1/α^2)

            if SCDiam in [:small, :large]
                #[7] (Interp5)
                @constraint(model, tr(G*(⊙(𝐳[:,i] - 𝐱[:,k], 𝐳[:,i] - 𝐱[:,k]))) <= (λ*D)^2)
            end
        end
    end

    #K,K Constraints
    if SCDiam in [:small, :large]
        for k in K
            for l in K
                if k==l
                    continue
                end
                #[8] (Interp6)
                @constraint(model, tr(G*(⊙(𝐱[:,k] - 𝐱[:,l], 𝐱[:,k] - 𝐱[:,l]))) <= (λ*D)^2)
            end
        end
    end

end


function setConstraints_SmoothSC(model, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, α, β; SCDiam = :large)

    γ = 1/(1/α - 1/β)

    if SCDiam == :large
        λ = 1
    elseif SCDiam == :small
        λ = 1/sqrt(2)
    end

    #I,I Constraints [Only involve g,z,n]
    for i in I
        for j in I
            if i==j
                continue
            end
            #[1] (Interp1)
            @constraint(model, tr(G*(⊙(𝐳[:,j]-1/β*𝐧[:,j]-𝐳[:,i]+1/α*𝐧[:,i],  𝐳[:,j]-1/β*𝐧[:,j]-𝐳[:,i]+1/α*𝐧[:,i]))) <= 1/γ^2 )  # Strongly convex constraint

            #[2] (Relaxation of n = -g/||g||)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐧[:,j] - 𝐧[:,i]))) <= 0)

            if SCDiam in [:small, :large]
            #[3] (Interp4)
                @constraint(model, tr(G*(⊙(𝐳[:,j]-1/β*𝐧[:,j]-𝐳[:,i]+1/β*𝐧[:,i], 𝐳[:,j]-1/β*𝐧[:,j]-𝐳[:,i]+1/β*𝐧[:,i] ))) <= (λ*(D - 2/β))^2 )
            end
        end
    end

    # I constraints [Only involve g,z,n]
    for i in I

        #[4] (Relaxation of n = -g/||g||)
        @constraint(model, tr(G*(⊙(𝐧[:,i], 𝐧[:,i]))) == 1)
        
        #[5] (Relaxation of n = -g/||g||)
        @constraint(model, tr(G*(⊙(𝐠[:,i], 𝐧[:,i]))) <= 0 )
    end

    #I,K constraints
    for i in I
        for k in K
            #***INCLUDE case where i=k, it is necessary***

            #[6] (Interp2)
            @constraint(model, tr(G*(⊙(𝐰[:,k] - 𝐳[:,i] + 1/α*𝐧[:,i], 𝐰[:,k] - 𝐳[:,i] + 1/α*𝐧[:,i]))) <= 1/γ^2 )

            if SCDiam in [:small, :large]
                #[7] (Interp5)
                @constraint(model, tr(G*(⊙(𝐰[:,k] - 𝐳[:,i] + 1/β*𝐧[:,i], 𝐰[:,k] - 𝐳[:,i] + 1/β*𝐧[:,i] ))) <= (λ*(D - 2/β))^2 )
            end
        end
    end

    #K constraints
    for k in K
        #[8] (Interp3)
        @constraint(model, tr(G*(⊙(𝐱[:,k] - 𝐰[:,k], 𝐱[:,k] - 𝐰[:,k]))) <= 1/β^2)
    end

    #K,K Constraints
    if SCDiam in [:small, :large]
        for k in K
            for l in K
                if k==l
                    continue
                end
                #[9] (Interp6)
                @constraint(model, tr(G*(⊙(𝐰[:,k]-𝐰[:,l], 𝐰[:,k]-𝐰[:,l] ))) <= (λ*(D - 2/β))^2 )
            end
        end
    end

end


function setConstraintsOptInterior(model, setType, D, I, K, G, 𝐱, 𝐠, 𝐳, 𝐧, 𝐰, α, β, δ; SCDiam = :large)

    if SCDiam == :large
        λ = 1
    elseif SCDiam == :small
        λ = 1/sqrt(2)
    end

    #Apply δ-Interior constraints
    if setType == :convex
        for i in I
            # x_⋆ (Interp2)
            @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐱[:,-1] + δ*𝐧[:,i] - 𝐳[:,i]))) <= 0)
            # x_⋆ (Interp5)
            @constraint(model, tr(G*(⊙(𝐳[:,i] - 𝐱[:,-1], 𝐳[:,i] - 𝐱[:,-1]))) <= (D-δ)^2)
        end
        for k in K
            # x_⋆ (Interp6)
            @constraint(model, tr(G*(⊙(𝐱[:,k] - 𝐱[:,-1], 𝐱[:,k] - 𝐱[:,-1]))) <= (D-δ)^2)
        end
    
    elseif setType == :SC
        
        if SCDiam == :large
            λ = 1
        elseif SCDiam == :small
            λ = 1/sqrt(2)
        end

        for i in I
            # x_⋆ (Interp2)
            @constraint(model, tr(G*(⊙(𝐱[:,-1]-𝐳[:,i]+1/α*𝐧[:,i], 𝐱[:,-1]-𝐳[:,i]+1/α*𝐧[:,i]))) <= (1/α - δ)^2)
            if SCDiam in [:large,:small]
                # x_⋆ (Interp5)
                @constraint(model, tr(G*(⊙(𝐳[:,i] - 𝐱[:,-1], 𝐳[:,i] - 𝐱[:,-1]))) <= (λ*D - δ)^2 )
            end
        end
        if SCDiam in [:large,:small]
            for k in K
                # x_⋆ (Interp6)
                @constraint(model, tr(G*(⊙(𝐱[:,k] - 𝐱[:,-1], 𝐱[:,k] - 𝐱[:,-1]))) <= (λ*D - δ)^2 )
            end
        end

    elseif setType in [:smooth, :smoothSC]
        if setType == :smoothSC
            if SCDiam == :large
                λ = 1
            elseif SCDiam == :small
                λ = 1/sqrt(2)
            end
        end

        γ = 1/(1/α - 1/β)
        s = max(0, δ - 1/β)

        for i in I
            # x_⋆ (Interp2)
            if setType == :smooth
                @constraint(model, tr(G*(⊙(-𝐠[:,i], 𝐰[:,-1] + s*𝐧[:,i] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= 0)
            elseif setType == :smoothSC
                @constraint(model, tr(G*(⊙(𝐰[:,-1] - 𝐳[:,i] + 1/α*𝐧[:,i], 𝐰[:,-1] - 𝐳[:,i] + 1/α*𝐧[:,i]))) <= (1/γ - s)^2 )
            end

            # x_⋆ (Interp5)
            @constraint(model, tr(G*(⊙(𝐰[:,-1] - 𝐳[:,i] + 1/β*𝐧[:,i], 𝐰[:,-1] - 𝐳[:,i] + 1/β*𝐧[:,i]))) <= (λ*(D-2/β)-s)^2 )
        end
    
        for k in K
            # x_⋆ (Interp6)
            @constraint(model, tr(G*(⊙(𝐰[:,k] - 𝐰[:,-1], 𝐰[:,k] - 𝐰[:,-1]))) <= (λ*(D-2/β)-s)^2 )
        end
    
        # x_⋆ (Interp3)
        @constraint(model, tr(G*(⊙(𝐰[:,-1] - 𝐱[:,-1], 𝐰[:,-1] - 𝐱[:,-1]))) <= (1/β - δ + s)^2)

    end

end


###-----Apply function interpolation constraints-----###
function setPrimalFuncConstraints(model, functionType, Ft, G, 𝐟, 𝐠, 𝐱, h, K, L, μ)
    
    #Smooth constraints
    #--------------------
    if functionType == :smooth
        for k in K, l in K
            if k != l
                # Smooth and Convex
                @constraint(model, Ft'*a_vec(k,l,𝐟) + tr(G*A_mat(k,l,𝐠,𝐱)) + ((1/(2*L))* tr(G*C_mat(k,l,𝐠))) <= 0)
            end
        end


    #Smooth, Strongly Convex constraints
    #----------------------
    elseif functionType == :smoothSC
        for k in K, l in K
            if k != l
                # Smooth and Strongly Convex
                @constraint(model, Ft'*a_vec(k,l,𝐟) + tr(G*A_mat(k,l,𝐠,𝐱)) + 1/(2*(1-μ/L))*(1/L*tr(G*C_mat(k,l,𝐠)) + μ*tr(G*B_mat(k,l,𝐱)) + 2*μ/L*(tr(G*A_mat(k,l,𝐠,𝐱)) + tr(G*A_mat(l,k,𝐠,𝐱)))  ) <= 0   )
            end
        end
    end

end
