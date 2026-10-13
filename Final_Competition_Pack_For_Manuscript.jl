# Agent-based competition on a staggered-column hexagonal lattice.
# Each matrix entry is 0 (empty) or an encoded cell: tens = cell-cycle age,
# ones digit = type (1 wild type, 2 mutant), fraction = persistent noise state.
# A time step grows cells, ages them, scores competition, removes cells, then
# shifts surviving cells into the resulting holes.
using Base.Threads
using Distributions
using Graphs
using SparseArrays
using Random
using LinearAlgebra
using CSV

# Extract the type at every site while discarding age and fractional noise.
function get_types(matrix)
    return floor.(matrix).%10
end

# Scalar version of get_types, for one encoded cell value.
function get_type(point)
    return floor(point)%10
end
# Euclidean distance after mapping matrix coordinates onto hexagon centers.
# The parity term accounts for the alternating offset of adjacent columns.
function hexdist(point1,point2)
    xdist = point1[1] - point2[1]
    ydist = point1[2] - point2[2]
    sqrt3 = sqrt(3)
    return sqrt((xdist*sqrt3 - ydist%2 * sqrt3/2)^2 + 2.25*ydist^2)
end

# Seed a centered wild-type patch with a smaller mutant patch inside it.
# Random initial ages and fractional states keep cells from dividing in sync.
function init_eyeball_track(r=101,c=101,size=minimum([r,c]),innersize=2,cyclelength1=39,cyclelength2=26)
    A = zeros(r,c)
    startr = round(Int, r/2 - size/2 + 1, RoundUp)
    startc = round(Int, c/2 - size/2 + 1, RoundUp)
    A[startr:startr+size-1,startc:startc+size-1].=1
    startinr = floor(Int,r/2)-floor(Int,innersize/2)+1
    startinc = floor(Int,c/2)-floor(Int,innersize/2)+1
    A[startinr:startinr+innersize-1,startinc:startinc+innersize-1].=2
    for i in 1:length(A[:,1])
        for j in 1:length(A[1,:])
            if A[i,j] != 0
                if A[i,j] == 1
                    A[i,j] = A[i,j] + rand() + rand(1:cyclelength1)*10
                elseif A[i,j] == 2
                    A[i,j] = A[i,j] + rand() + rand(1:cyclelength2)*10
                end
            end
        end
    end
    return A
end

# Return cells of the requested type whose encoded age reaches its cycle length.
function choosecells_tracked(matrix,celltype,cyclelength)
    typemat = get_types(matrix)
    livecells = findall(x->x==celltype,typemat)
    locations = []
    for i in livecells
        if round(matrix[i],digits=-1)/10 >= cyclelength
            push!(locations,i)
        end
    end
    return locations
end

# Search expanding square perimeters for empty sites, then choose at random
# among the closest sites by hexagonal distance. Return nothing if none exists.
function findhole(matrix,initialcoords)
    line1 = copy(initialcoords[1]) #line going ^
    line3 = copy(initialcoords[1]) #line going v
    line2 = copy(initialcoords[2]) #line going >
    line4 = copy(initialcoords[2]) #line going <
    
    closest = []
    # Square search rings need not be ordered by hex distance. Continue
    # briefly after a hit so a later ring can still contain a closer site.
    hit = 10
    number = 1
    while hit > 0 && number < length(matrix[1,:])*2
        found = []
        number+=1
        if line1+1 <= length(matrix[:,1]) #one step ^ if doesn't exceed bounds
            line1+=1
        end
        if line2+1 <= length(matrix[1,:]) #one step >
            line2+=1
        end
        if line3-1 > 0 #one step v
            line3-=1
        end
        if line4-1 > 0 #one step <
            line4-=1
        end

        #check if 0 in line1
        for i in line4:line2
            if matrix[line1,i] == 0
                push!(found,[line1,i])
            end
        end

        #check if 0 in line3
        for i in line4:line2
            if matrix[line3,i] == 0
                push!(found,[line3,i])
            end
        end

        #check if 0 in line2 (exclude already counted by line1 and line3)
        for i in line3+1:line1-1
            if matrix[i,line2] == 0
                push!(found,[i,line2])
            end
        end

        #check if 0 in line 4 (exclude already counted by line1 and line3)
        for i in line3+1:line1-1
            if matrix[i,line4] == 0
                push!(found,[i,line4])
            end
        end

        #if found a 0, do something
        if length(found) != 0
            if hit != 10
                for j in found
                    distdiff = hexdist(initialcoords,j)-hexdist(initialcoords,closest[1])
                    if distdiff < 0
                        closest = [j]
                        hit = 4
                    elseif distdiff == 0
                        push!(closest,j)
                    end
                end
            else
                if length(closest)==0
                    push!(closest,found[1])
                    found = copy(found[2:end])
                end
                for j in found
                    distdiff = hexdist(initialcoords,j)-hexdist(initialcoords,closest[1])
                    if distdiff < 0
                        closest = [j]
                    elseif distdiff == 0
                        push!(closest,j)
                    end
                end   
            end
            hit-=1
        end
    end

    if length(closest) == 0
        return nothing
    end

    return sample(closest)
end

# Build the fixed hex-lattice graph; vertex IDs use Julia's column-major order.
# Cell-type-specific edges are removed from copies of this graph during growth.
function get_adjacencymatrix(matrix,r,c,nbs)
    adjmat = zeros(r*c,r*c)
    for i in 1:length(matrix[:,1])
        for j in 1:length(matrix[1,:])
            maincellid = i+r*(j-1)
            mneighbors = nbs[[i,j]]
            for k in 1:length(mneighbors[:,1])
                mneighborid = mneighbors[k,1]+r*(mneighbors[k,2]-1)
                adjmat[maincellid,mneighborid] = 1
            end
        end
    end
    ag=SimpleGraph(sparse(adjmat .!= 0))
    return ag
end

# Map each [row, column] coordinate to its in-bounds hexagonal neighbors.
# Neighbor offsets differ for even and odd columns of the staggered lattice.
function get_neighbors(matrix)
    
    rowlen = length(matrix[:,1])
    collen = length(matrix[1,:])
    evenmat = [1 -1;1 0;1 1;0 -1;0 1;-1 0]
    oddmat = [-1 -1;1 0;-1 1;0 -1;0 1;-1 0]
    ndict = Dict()
    for j in 1:collen
        if iseven(j)
            for i in 1:rowlen
                neighbors = evenmat .+ [i j]
                if i==1 || j==1 || i==rowlen || j==collen
                    removals = [true;true;true;true;true;true]
                    for n in 1:6
                        if neighbors[n,1] < 1 || neighbors[n,1] > rowlen || neighbors[n,2] < 1 || neighbors[n,2] > collen
                            removals[n] = false
                        end
                    end
                    neighbors = neighbors[removals,:]
                end
                ndict[[i,j]] = neighbors
            end
        else
            for i in 1:rowlen
                neighbors = oddmat .+ [i j]
                if i==1 || j==1 || i==rowlen || j==collen
                    removals = [true;true;true;true;true;true]
                    for n in 1:6
                        if neighbors[n,1] < 1 || neighbors[n,1] > rowlen || neighbors[n,2] < 1 || neighbors[n,2] > collen
                            removals[n] = false
                        end
                    end
                    neighbors = neighbors[removals,:]
                end
                ndict[[i,j]] = neighbors
            end
        end
    end
    return ndict
end

# Main growth rule. Process eligible cells in random order, push a connected
# chain toward an empty site, then place a daughter next to the dividing cell.
# Mutants use a tissue-edge hole and may route through mutant and wild-type
# regions separately; the final pass moves disconnected boundary islands.
# Mutates and returns matrix. ag is the prebuilt graph of all lattice sites.
function replicate3_tracked_pathfinder_faster_redshoverule_movebubbles_edgefinder(cyclelength1,cyclelength2,matrix,daughterdrift,r,c,ag)
    #choose cells to replicate
    reps = shuffle(vcat(choosecells_tracked(matrix,1,cyclelength1),choosecells_tracked(matrix,2,cyclelength2)))
    
    betadist = Beta(1,5)
    
    nbs = get_neighbors(matrix)
    
    

    #step through each cell 
    for rep in reps
        typemat = get_types(matrix)

        # Rebuild type-specific paths after each division, since earlier pushes
        # may have changed which sites contain each cell type.
        agtemp = deepcopy(ag)
        bad = []
        for i in eachindex(matrix)
            if typemat[i] != 1
                push!(bad,i)
            end
        end
        for v in bad
            nbrs = collect(neighbors(agtemp, v)) # snapshot; neighbors() changes as we delete
            for u in nbrs
                rem_edge!(agtemp, u, v)
            end
        end

        # The second graph retains edges only between mutant sites.
        agtemp2 = deepcopy(ag)
        bad2 = []
        for i in eachindex(matrix)
            if typemat[i] != 2
                push!(bad2,i)
            end
        end
        for v in bad2
            nbrs = collect(neighbors(agtemp2, v)) # snapshot; neighbors() changes as we delete
            for u in nbrs
                rem_edge!(agtemp2, u, v)
            end
        end

        #find nearest hole
        #hole = findhole_random(get_types(matrix))
        if typemat[rep[1],rep[2]] == 1
            hole = findhole(typemat,rep)
        else
            hole = edgefinder_hole(typemat)
        end
        if hole===nothing
            return matrix
            #if there are no holes, the cell does not replicate
        end

        # A* ends on an occupied neighbor; the empty site is appended afterward.
        hole_neighbors = nbs[hole]
        hole_adjacents = []
        for i in 1:length(hole_neighbors[:,1])
            if typemat[hole_neighbors[i,1],hole_neighbors[i,2]] != 0
                push!(hole_adjacents,[hole_neighbors[i,1],hole_neighbors[i,2]])
            end
        end
        if isempty(hole_adjacents)
            found_new_hole = false

            for i in axes(hole_neighbors, 1)
                # Empty site between the tissue and the original distant hole
                intermediate = [
                    hole_neighbors[i,1],
                    hole_neighbors[i,2]
                ]

                intermediate_neighbors = nbs[intermediate]

                for j in axes(intermediate_neighbors, 1)
                    occupied_neighbor = [
                        intermediate_neighbors[j,1],
                        intermediate_neighbors[j,2]
                    ]

                    if typemat[occupied_neighbor...] == 1
                        # Grow into the intermediate site, not the distant hole
                        hole = intermediate

                        # A* must terminate at an occupied graph vertex
                        push!(hole_adjacents, occupied_neighbor)

                        found_new_hole = true
                        break
                    end
                end

                found_new_hole && break
            end
        end

        if isempty(hole_adjacents)
            continue
        end
        hole_adjacent = rand(hole_adjacents)
        
        #find shortest path
        repid = rep[1]+r*(rep[2]-1)
        holeid = hole_adjacent[1]+r*(hole_adjacent[2]-1)
        if repid==holeid
            path = []
        else
            if typemat[rep[1],rep[2]] == 1
                edges_path = a_star(agtemp,repid,holeid)

                if isempty(edges_path)
                    #println("edges_path empty, cell type 1")
                    #println(repid)
                    #println(holeid)
                    continue
                end

                pathids = []
                for i in edges_path
                    if dst(i) ∉ pathids
                        push!(pathids,dst(i))
                    end
                end
                path = []
                for i in pathids
                    push!(path,[Int((i-1)%r+1),Int(floor((i-1)/r)+1)])
                end
            else
                repneighbors = nbs[[rep[1],rep[2]]]
                gneighbor = []
                for i in 1:length(repneighbors[:,1])
                    if typemat[repneighbors[i,1],repneighbors[i,2]] == 1
                        #### MIGHT NOT NEED TO DO THIS IF ALL BUBBLES ARE PUSHED TO THE EDGE
                        #if repneighbors[i,1] + r*(repneighbors[i,2]-1) ∈ main_comp
                        push!(gneighbor,[repneighbors[i,1],repneighbors[i,2]])
                    end
                end
                if length(gneighbor) == 0
                    # Reach the wild-type path through a randomly chosen
                    # mutant/wild-type boundary when no wild type touches rep.
                    redcells = findall(x->x==2,typemat)
                    redboundary = []
                    ### CAN USE EDGEFINDER
                    for j in redcells
                        neighborcells = nbs[[j[1],j[2]]]
                        for k in 1:length(neighborcells[:,1])
                            if typemat[neighborcells[k,1],neighborcells[k,2]] == 1
                                push!(redboundary,[[neighborcells[k,1],neighborcells[k,2]],[j[1],j[2]]])
                            end
                        end
                    end
                    redbound = rand(redboundary)
                    rbid = redbound[2][1]+r*(redbound[2][2]-1)
                    gbid = redbound[1][1]+r*(redbound[1][2]-1)

                    edges_path1 = a_star(agtemp2,repid,rbid)
                    if isempty(edges_path1)
                        #println("edges_path1 empty, no gneighbors")
                        continue
                    end
                    edges_path2 = a_star(agtemp,gbid,holeid)
                    if isempty(edges_path2)
                        #println("edges_path2 empty, no gneighbors")
                        continue
                    end
                    pathids = []
                    for i in edges_path1
                        if dst(i) ∉ pathids
                            push!(pathids,dst(i))
                        end
                    end
                    push!(pathids,gbid)
                    for i in edges_path2
                        if dst(i) ∉ pathids
                            push!(pathids,dst(i))
                        end
                    end
                else
                    # A neighboring wild-type cell provides a shorter bridge.
                    gselect = rand(gneighbor)
                    gid = gselect[1]+r*(gselect[2]-1)

                    edges_path = a_star(agtemp,gid,holeid)
                    if isempty(edges_path)
                        continue
                    end
                    pathids = []
                    push!(pathids,gid)
                    for i in edges_path
                        if dst(i) ∉ pathids
                            push!(pathids,dst(i))
                        end
                    end
                end

                path = []
                for i in pathids
                    push!(path,[Int((i-1)%r+1),Int(floor((i-1)/r)+1)])
                end
            end
        end
        push!(path,hole)

        # Shift cells from the hole backward so no occupied value is overwritten.
        for i in reverse(1:length(path))
            if i != length(path)
                matrix[path[i+1][1],path[i+1][2]] = matrix[path[i][1],path[i][2]]
            end
        end

        # The daughter inherits some fractional state, draws a new age, and
        # replaces the first displaced site; the parent resets its cycle age.
        div_cell = matrix[rep[1],rep[2]]
        div_cell_type= typemat[rep[1],rep[2]]
        if div_cell_type==1
            matrix[path[1][1],path[1][2]] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
        elseif div_cell_type==2
            matrix[path[1][1],path[1][2]] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
        end
        matrix[rep[1],rep[2]] = div_cell%10
    end

    
    # Group wild-type cells bordering mutants into boundary components.
    # Swap cells from smaller components toward the longest boundary.
    candidates = edgefinder_mutboundary(matrix)
    if isempty(candidates)
        return matrix
    end
    lines = build_lines(candidates, nbs)
    longest_idx = argmax(length.(lines))
    longest = argmax(length,lines)
    islands = [x for (i, line) in enumerate(lines) if i != longest_idx for x in line]
    dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    for i in islands
        possible_directions = findall(x->x[1]==i[1]||x[2]==i[2],longest)
        if length(possible_directions) == 0
            direction = rand(dirs)
        else
            #closest_idx = argmin(norm.(Tuple.(longest[possible_directions] .- i)))
            closest_idx = argmin([norm(Tuple(p - i)) for p in longest[possible_directions]])
            if length(closest_idx) > 1
                closest_idx = closest_idx[1]
            end
            dindicator = longest[possible_directions][closest_idx] - i
            if dindicator[1] == 0
                if dindicator[2] > 0
                    direction = dirs[3]
                else
                    direction = dirs[4]
                end
            elseif dindicator[2] == 0
                if dindicator[1] > 0 
                    direction = dirs[1]
                else
                    direction = dirs[2]
                end
            end
        end
        chosen_cell = [i[1],i[2]] .+ direction
        moving_cell = matrix[i[1],i[2]]
        shoved_cell = matrix[chosen_cell[1],chosen_cell[2]]
        matrix[i[1],i[2]] = shoved_cell
        matrix[chosen_cell[1],chosen_cell[2]] = moving_cell
    end

    return matrix
end

# Initialization growth rule: push along one or two hexagonal axes toward the
# nearest hole. This geometric version is used before the main graph-based run.
# Mutates and returns matrix.
function replicate3_tracked(cyclelength1,cyclelength2,matrix,daughterdrift)
    #choose cells to replicate
    reps = shuffle(vcat(choosecells_tracked(matrix,1,cyclelength1),choosecells_tracked(matrix,2,cyclelength2)))
    
    betadist = Beta(1,5)
    #step through each cell 
    for rep in reps
        #find nearest hole
        #hole = findhole_random(get_types(matrix))
        hole = findhole(get_types(matrix),rep)
        if hole===nothing
            return matrix
            #if there are no holes, the cell does not replicate
        end
        
        # Angle selects either a straight hex-axis push or a two-axis route.
        # For two-axis routes, choose the order randomly where both fit.
        ang = atan((((rep[1]-hole[1])*sqrt(3))+sqrt(3)/2*((-1)^((hole[2])+1))*(abs(rep[2]-hole[2])%2))/((rep[2]-hole[2])*1.5))
        
        #if line between hole and rep align with one of the axes
        #if ang in [-pi/2,-pi/6,pi/6,pi/2]
        if round(ang,digits=7) in [-1.5707963,-0.5235988,0.5235988,1.5707963]
            #if ang in [-pi/2,pi/2]
            if round(ang,digits=7) in [-1.5707963,1.5707963]
                a = rep[1]
                b = hole[1]
                foba = a > b ? 1 : -1
                for i in foba:foba:a-b-foba
                    matrix[hole[1]+i-foba,hole[2]] = matrix[hole[1]+i,hole[2]]
                end
                div_cell = matrix[rep[1],rep[2]]
                div_cell_type=get_type(div_cell)
                if div_cell_type==1
                    matrix[rep[1]-foba,rep[2]] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                elseif div_cell_type==2
                    matrix[rep[1]-foba,rep[2]] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                end
                matrix[rep[1],rep[2]] = div_cell%10
            #elseif ang == pi/6
            elseif round(ang,digits=7) == 0.5235988
                a = rep[2]
                b = hole[2]
                foba = a > b ? 1 : -1
                for i in foba:foba:a-b-foba
                    matrix[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = matrix[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                end
                div_cell = matrix[rep[1],rep[2]]
                div_cell_type=get_type(div_cell)
                if div_cell_type==1
                    matrix[Int(floor((rep[2]-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                elseif div_cell_type==2
                    matrix[Int(floor((rep[2]-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                end
                matrix[rep[1],rep[2]] = div_cell%10
            else
                a = rep[2]
                b = hole[2]
                foba = a > b ? 1 : -1
                for i in foba:foba:a-b-foba
                    matrix[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = matrix[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                end
                div_cell = matrix[rep[1],rep[2]]
                div_cell_type=get_type(div_cell)
                if div_cell_type==1
                    matrix[Int(-floor((rep[2]-foba)/2)+hole[1]+floor((hole[2])/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                elseif div_cell_type==2
                    matrix[Int(-floor((rep[2]-foba)/2)+hole[1]+floor((hole[2])/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                end
                matrix[rep[1],rep[2]] = div_cell%10
            end

        #in all other cases, find the two axes that shoving will occur on
        else
            if -pi/2<ang<-pi/6
                ax1 = 1
                ax2 = 3
                direc = rand([ax1,ax2])
                if direc == ax1
                    #if ax1 == 1
                    a = rep[1] #replicating cell y-coord
                    b = hole[1] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = hole[2]
                    yintersect = Int(-floor((xintersect)/2)+rep[1]+floor((rep[2])/2))
                
                    for i in foba:foba:yintersect-b
                        matrix[hole[1]+i-foba,hole[2]] = matrix[hole[1]+i,hole[2]]
                    end
                
                    c = rep[2] #replicating cell x-coord
                    d = hole[2] #hole x-coord
                    foba2 = c > d ? 1 : -1
                
                    #yax = floor((xax+1)/2)+ya-floor((xa+1)/2)
                    for j in foba2:foba2:c-d-foba2
                        matrix[Int(-floor((hole[2]+j-foba2)/2)+rep[1]+floor((rep[2])/2)),hole[2]+j-foba2] = matrix[Int(-floor((hole[2]+j)/2)+rep[1]+floor((rep[2])/2)),hole[2]+j]
                    end
                    #replicating cell
                    div_cell = matrix[rep[1],rep[2]]
                    div_cell_type=get_type(div_cell)
                    if div_cell_type==1
                        matrix[Int(-floor((hole[2]+c-d-foba2)/2)+rep[1]+floor((rep[2])/2)),hole[2]+c-d-foba2] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                    elseif div_cell_type==2
                        matrix[Int(-floor((hole[2]+c-d-foba2)/2)+rep[1]+floor((rep[2])/2)),hole[2]+c-d-foba2] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                    end
                    matrix[rep[1],rep[2]] = div_cell%10
                
                else
                    #if ax1 == 1
                    a = rep[2] #replicating cell x-coord
                    b = hole[2] #hole x-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = rep[2]
                    yintersect = Int(-floor((xintersect)/2)+hole[1]+floor((hole[2])/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = matrix[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                    end
                
                    c = rep[1] #replicating cell y-coord
                    d = yintersect #hole y-coord
                    foba2 = c > d ? 1 : -1
                
                    for j in foba2:foba2:c-d-foba2
                        matrix[yintersect+j-foba2,xintersect] = matrix[yintersect+j,xintersect]
                    end
                    #replicating cell
                    div_cell = matrix[rep[1],rep[2]]
                    div_cell_type=get_type(div_cell)
                    if div_cell_type==1
                        matrix[c-foba2,xintersect] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                    elseif div_cell_type==2
                        matrix[c-foba2,xintersect] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                    end
                    matrix[rep[1],rep[2]] = div_cell%10
                end
            elseif -pi/6<ang<pi/6
                ax1 = 2
                ax2 = 3
                #check to see if shoving could reach above or below dimensions of matrix (in which case, switches to start with other axis)
                ax1intersect = Int(floor((Int(floor((hole[2]+1)/2)+floor((rep[2])/2)+rep[1]-hole[1])+1)/2)+hole[1]-floor((hole[2]+1)/2))
                ax2intersect = Int(floor((Int(floor((rep[2]+1)/2)+floor((hole[2])/2)+hole[1]-rep[1])+1)/2)+rep[1]-floor((rep[2]+1)/2))
                if ax1intersect > length(matrix[:,1]) || ax1intersect < 1
                    direc = ax2
                elseif ax2intersect > length(matrix[:,1]) || ax2intersect < 1
                    direc = ax1
                else
                    direc = rand([ax1,ax2])
                end
                if direc == ax1
                    #if ax1 == 1
                    a = rep[2] #replicating cell y-coord
                    b = hole[2] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = Int(floor((hole[2]+1)/2)+floor((rep[2])/2)+rep[1]-hole[1])
                    yintersect = Int(floor((xintersect+1)/2)+hole[1]-floor((hole[2]+1)/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = matrix[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                    end
                
                    for j in foba:foba:a-xintersect-foba
                        matrix[Int(-floor((xintersect+j-foba)/2)+rep[1]+floor((rep[2])/2)),xintersect+j-foba] = matrix[Int(-floor((xintersect+j)/2)+rep[1]+floor((rep[2])/2)),xintersect+j]
                    end
                    #replicating cell
                    div_cell = matrix[rep[1],rep[2]]
                    div_cell_type=get_type(div_cell)
                    if div_cell_type==1
                        matrix[Int(-floor((rep[2]-foba)/2)+rep[1]+floor((rep[2])/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                    elseif div_cell_type==2
                        matrix[Int(-floor((rep[2]-foba)/2)+rep[1]+floor((rep[2])/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                    end
                    matrix[rep[1],rep[2]] = div_cell%10
                else
                    #if ax1 == 1
                    a = rep[2] #replicating cell y-coord
                    b = hole[2] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = Int(floor((rep[2]+1)/2)+floor((hole[2])/2)+hole[1]-rep[1])
                    yintersect = Int(floor((xintersect+1)/2)+rep[1]-floor((rep[2]+1)/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = matrix[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                    end
                
                    for j in foba:foba:a-xintersect-foba
                        matrix[Int(floor((xintersect+j-foba+1)/2)+rep[1]-floor((rep[2]+1)/2)),xintersect+j-foba] = matrix[Int(floor((xintersect+j+1)/2)+rep[1]-floor((rep[2]+1)/2)),xintersect+j]
                    end
                    #replicating cell
                    div_cell = matrix[rep[1],rep[2]]
                    div_cell_type=get_type(div_cell)
                    if div_cell_type==1
                        matrix[Int(floor((rep[2]-foba+1)/2)+rep[1]-floor((rep[2]+1)/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                    elseif div_cell_type==2
                        matrix[Int(floor((rep[2]-foba+1)/2)+rep[1]-floor((rep[2]+1)/2)),rep[2]-foba] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                    end
                    matrix[rep[1],rep[2]] = div_cell%10
                end
            elseif pi/6<ang<pi/2
                ax1 = 1
                ax2 = 2
                direc = rand([ax1,ax2])
                if direc == ax1
                    #if ax1 == 1
                    a = rep[1] #replicating cell y-coord
                    b = hole[1] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = hole[2]
                    yintersect = Int(floor((xintersect+1)/2)+rep[1]-floor((rep[2]+1)/2))
                
                    for i in foba:foba:yintersect-b
                        matrix[hole[1]+i-foba,hole[2]] = matrix[hole[1]+i,hole[2]]
                    end
                
                    c = rep[2] #replicating cell x-coord
                    d = hole[2] #hole x-coord
                    foba2 = c > d ? 1 : -1
                
                    #yax = floor((xax+1)/2)+ya-floor((xa+1)/2)
                    for j in foba2:foba2:c-d-foba2
                        matrix[Int(floor((hole[2]+j-foba2+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+j-foba2] = matrix[Int(floor((hole[2]+j+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+j]
                    end
                    #replicating cell
                    div_cell = matrix[rep[1],rep[2]]
                    div_cell_type=get_type(div_cell)
                    if div_cell_type==1
                        matrix[Int(floor((hole[2]+c-d-foba2+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+c-d-foba2] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                    elseif div_cell_type==2
                        matrix[Int(floor((hole[2]+c-d-foba2+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+c-d-foba2] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                    end
                    matrix[rep[1],rep[2]] = div_cell%10
                else
                    #if ax1 == 1
                    a = rep[2] #replicating cell x-coord
                    b = hole[2] #hole x-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = rep[2]
                    yintersect = Int(floor((xintersect+1)/2)+hole[1]-floor((hole[2]+1)/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = matrix[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                    end
                
                    c = rep[1] #replicating cell y-coord
                    d = yintersect #hole y-coord
                    foba2 = c > d ? 1 : -1
                
                    yax = floor((hole[2]+yintersect-b+1)/2)+hole[1]-floor((hole[2]+1)/2)
                    for j in foba2:foba2:c-d-foba2
                        matrix[yintersect+j-foba2,xintersect] = matrix[yintersect+j,xintersect]
                    end
                    #replicating cell
                    div_cell = matrix[rep[1],rep[2]]
                    div_cell_type=get_type(div_cell)
                    if div_cell_type==1
                        matrix[c-foba2,xintersect] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength1*10,digits=-1)
                    elseif div_cell_type==2
                        matrix[c-foba2,xintersect] = div_cell_type + daughterdrift*(div_cell-floor(div_cell)) + (1-daughterdrift)*rand() + round(rand(betadist)*cyclelength2*10,digits=-1)
                    end
                    matrix[rep[1],rep[2]] = div_cell%10
                end
            end
        end
    end
    return matrix;
end

# Partition candidate boundary cells into connected components using the
# precomputed hex-neighbor map. Each returned line is a vector of indices.
function build_lines(candidates, nbs)
    # Make a fast membership structure for "is this a candidate?"
    candidate_set = Set(candidates)

    visited = Set{CartesianIndex{2}}()
    lines = Vector{Vector{CartesianIndex{2}}}()  # each "line" is a Vector of indices

    for c in candidates
        c in visited && continue  # already assigned to a line

        # Start new component (line) from c
        line  = CartesianIndex{2}[]
        queue = CartesianIndex{2}[c]
        push!(visited, c)

        while !isempty(queue)
            curr = pop!(queue)    # BFS/DFS, doesn't matter much
            push!(line, curr)

            # Look up neighbors for this cell
            bob = nbs[[curr[1], curr[2]]]   # 6×2 or fewer

            @inbounds for r in eachrow(bob)
                nb = CartesianIndex(r[1], r[2])

                # Only step to neighbors that are also candidates and not visited
                if nb in candidate_set && !(nb in visited)
                    push!(visited, nb)
                    push!(queue, nb)
                end
            end
        end

        push!(lines, line)
    end

    return lines
end

# Select wild-type sites at the mutant interface using the sum of their six
# neighbor types. Out-of-bounds neighbors retain the site's own type.
function edgefinder_mutboundary(matrix)
    nrow, ncol = size(matrix)

    # Assuming get_types is elementwise: type_of_cell.(matrix)
    types = get_types(matrix)

    # Start with self plus six neighbor slots. Each in-bounds neighbor
    # replaces one self contribution; missing neighbors keep that value.
    bigarmo = 7 .* types

    @inbounds for j in 1:ncol
        jparity    = j % 2          # j%2
        jp1parity  = (j + 1) % 2    # (j+1)%2

        for i in 1:nrow
            self = types[i, j]

            # armo1: i < nrow ⇒ neighbor (i+1, j)
            if i < nrow
                bigarmo[i, j] += types[i + 1, j] - self

                # armo3: i < nrow && j < ncol ⇒ (i + (j+1)%2, j+1)
                if j < ncol
                    bigarmo[i, j] += types[i + jp1parity, j + 1] - self
                end

                # armo4: i < nrow && j > 1 ⇒ (i + (j+1)%2, j-1)
                if j > 1
                    bigarmo[i, j] += types[i + jp1parity, j - 1] - self
                end
            end

            # armo2: i > 1 ⇒ neighbor (i-1, j)
            if i > 1
                bigarmo[i, j] += types[i - 1, j] - self

                # armo5: i > 1 && j < ncol ⇒ (i - j%2, j+1)
                if j < ncol
                    bigarmo[i, j] += types[i - jparity, j + 1] - self
                end

                # armo6: i > 1 && j > 1 ⇒ (i - j%2, j-1)
                if j > 1
                    bigarmo[i, j] += types[i - jparity, j - 1] - self
                end
            end
        end
    end

    # Collect candidate indices in a single pass:
    candidates = CartesianIndex[]
    @inbounds for I in CartesianIndices(matrix)
        # Retain wild-type sites whose encoded neighbor sum passes the
        # interface test used by the bubble-moving growth rule.
        if types[I] == 1 && bigarmo[I] % 7 != 0 && 7 < bigarmo[I]
            push!(candidates, I)
        end
    end

    return candidates
end

# Pick a random empty site next to occupied tissue by the same neighbor-sum
# encoding. Return nothing when there is no qualifying edge hole.
function edgefinder_hole(matrix)
    nrow, ncol = size(matrix)

    # Assuming get_types is elementwise: type_of_cell.(matrix)
    types = get_types(matrix)

    # Use the same self-plus-six-neighbor encoding as edgefinder_mutboundary.
    bigarmo = 7 .* types

    @inbounds for j in 1:ncol
        jparity    = j % 2          # j%2
        jp1parity  = (j + 1) % 2    # (j+1)%2

        for i in 1:nrow
            self = types[i, j]

            # armo1: i < nrow ⇒ neighbor (i+1, j)
            if i < nrow
                bigarmo[i, j] += types[i + 1, j] - self

                # armo3: i < nrow && j < ncol ⇒ (i + (j+1)%2, j+1)
                if j < ncol
                    bigarmo[i, j] += types[i + jp1parity, j + 1] - self
                end

                # armo4: i < nrow && j > 1 ⇒ (i + (j+1)%2, j-1)
                if j > 1
                    bigarmo[i, j] += types[i + jp1parity, j - 1] - self
                end
            end

            # armo2: i > 1 ⇒ neighbor (i-1, j)
            if i > 1
                bigarmo[i, j] += types[i - 1, j] - self

                # armo5: i > 1 && j < ncol ⇒ (i - j%2, j+1)
                if j < ncol
                    bigarmo[i, j] += types[i - jparity, j + 1] - self
                end

                # armo6: i > 1 && j > 1 ⇒ (i - j%2, j-1)
                if j > 1
                    bigarmo[i, j] += types[i - jparity, j - 1] - self
                end
            end
        end
    end

    # Collect candidate indices in a single pass:
    candidates = CartesianIndex[]
    @inbounds for I in CartesianIndices(matrix)
        # Empty sites with a nonzero neighbor sum modulo seven are candidates.
        if matrix[I] == 0 && bigarmo[I] % 7 != 0
            push!(candidates, I)
        end
    end

    if isempty(candidates)
        return nothing
    else
        choice = rand(candidates)
        return [choice[1],choice[2]]
    end
end

# Advance every occupied cell by one encoded cycle step (ten units).
function age_cells(matrix)
    agedmatrix = ifelse.(matrix .== 0, 0, matrix .+ 10)
    return agedmatrix
end

# Decode each cell's stored uniform quantile into a normal stress value.
# Empty sites remain zero.
function get_norm_smatrix_tracked(matrix,std)
    newmat = ifelse.(floor.(matrix) .== 0, 0, uni_to_norm.(matrix-floor.(matrix),get_types(matrix),std))
    return newmat
end

# Quantile transform: uniform state -> Normal(mean = cell type, std).
function uni_to_norm(uniform_draw,m,std)
    return quantile(Normal(m,std),[uniform_draw])[1]
end

# Alternative stress update: blend the previous stress with a fresh draw.
function new_smatrix_tracked(matrix,smatrix,std,rate)
    newstress = ifelse.(floor.(matrix) .== 0, 0, cellstress.(get_types(matrix),std))
    newsmatrix = smatrix.*(1-rate).+newstress.*rate
    return newsmatrix
end

# Autoregressive stress update. rate retains the previous deviation from the
# type mean; the scaled innovation keeps its variance at std squared.
function new_smatrix_tracked_noiseratefix(matrix,smatrix,std,rate)
    newstress = ifelse.(floor.(matrix) .==0, 0, cellstress.(0,1))
    newsmatrix = get_types(matrix) .+ rate.*(smatrix .- get_types(matrix)) .+ std*sqrt(1-rate^2).*newstress
    return newsmatrix
end

# Draw one normal stress sample with the requested mean and spread.
function cellstress(mean,std)
    N = Normal(mean,std)
    draw = rand(N)
    return draw
end

# Encode updated normal stress back into the fractional uniform quantile.
# The integer part, including type and age, is preserved.
function get_uni_matrix_tracked(matrix,smatrix,std)
    newmat = ifelse.(floor.(matrix) .== 0, 0, floor.(matrix).+norm_to_uni.(smatrix,get_types(matrix),std))
    return newmat
end

# Inverse transform used by get_uni_matrix_tracked.
function norm_to_uni(norm_draw,m,std)
    return cdf(Normal(m,std),norm_draw)
end

# Signed Hill response for competition magnitude |x|. Positive and negative
# scores use separate maximum responses; a zero score returns zero.
function comphill(x,pmax1,pmax2,n,k)
    if x > 0
        return pmax1*(abs(x)^n/(k^n+abs(x)^n))
    elseif x < 0
        return pmax2*(abs(x)^n/(k^n+abs(x)^n))
    else
        return 0
    end
end

# Update persistent cell noise, compare each live cell with its hex neighbors,
# and remove cells whose Hill response exceeds death_thresh. Empty neighbors
# contribute a fixed stress value of 1. Returns the updated matrix, stress and
# score maps, death counts, neighbor composition counts, stress count, and holes.
# This routine does not apply the regional stress-dependent second death pass.
function competition10_nostress_tracked(matrix,death1,death2,n,k,sigma,noiserate,death_thresh,stress_thresh1,stress_thresh2)
    #create indices for all cells
    cells = findall(x->x!=0,floor.(matrix))
    #create matrix for quickly finding neighbors
    neighbors=get_neighbors(matrix)
    #noise modification parameters
    #n = 3.5
    pmax1 = death1
    pmax2 = death2
    #k = 0.1

    #previous stress matrix
    oldsmatrix = get_norm_smatrix_tracked(matrix,sigma)
    # Retain part of the previous stress deviation and add a fresh noise draw.
    smatrix = new_smatrix_tracked_noiseratefix(matrix,oldsmatrix,sigma,noiserate)
    #convert new stress matrix into uniform distribution values (from normal values)
    newmatrix = get_uni_matrix_tracked(matrix,smatrix,sigma)

    matrixcopy = deepcopy(newmatrix)

    #matrix to store dead cell positions
    deads = []

    competition_matrix = zeros(length(matrix[:,1]),length(matrix[1,:]))

    reddead = 0
    greendead = 0
    # Count cells by their own type and their number of mutant neighbors.
    comptypedict = Dict(1=>Dict(0=>0,1=>0,2=>0,3=>0,4=>0,5=>0,6=>0),2=>Dict(0=>0,1=>0,2=>0,3=>0,4=>0,5=>0,6=>0))
    comptypemat = zeros(length(matrix[:,1]),length(matrix[1,:]))

    stress_curr = 0

    for i in cells
        ineighbors = neighbors[[i[1],i[2]]]
        neighborstress = []
        numred = 0
        #newneighbor = farcell_fastest([i[1],i[2]],matrix)
        for j in 1:length(ineighbors[:,1])
            if floor(matrix[ineighbors[j,1],ineighbors[j,2]]) != 0
                push!(neighborstress,smatrix[ineighbors[j,1],ineighbors[j,2]])
                if get_type(matrix[ineighbors[j,1],ineighbors[j,2]]) == 2
                    numred += 1
                end
            else
                #newneighbor = findfarthestcell_fast(matrix,[ineighbors[j,1],ineighbors[j,2]],neighbors)
                #push!(neighborstress,smatrix[newneighbor[1],newneighbor[2]])
                #if floor(matrix[newneighbor[1],newneighbor[2]]) == 2
                #    numred += 1
                #end
                push!(neighborstress,1)
            end
        end

        comptypedict[get_type(matrix[i])][numred] += 1
        comptypemat[i] = numred

        #death_thresh = 0.049
        
        if length(neighborstress) == 0
            nothing
        else
            competition_score = smatrix[i] - mean(neighborstress)
            competition_matrix[i] = competition_score
            #if abs(competition_score) > stress_thresh
            if competition_score > stress_thresh1 || competition_score < stress_thresh2
                stress_curr += 1
            end
            death = comphill(competition_score,pmax1,pmax2,n,k)
            #if abs(competition_score) > .8
            # Death is a fixed threshold test here, not a probability draw.
            if death > death_thresh
            #if rand() < death
                matrixcopy[i] = 0
                push!(deads,i)
                if get_type(matrix[i]) == 2
                    #=if numred == 6
                        collateral += 1
                    end=#
                    reddead += 1
                else
                    #=if numred == 0
                        collateral += 1
                    end=#
                    greendead += 1
                end      
            end
        end
    end

    return matrixcopy, smatrix, competition_matrix, reddead, comptypedict, stress_curr, deads, greendead
end

# Integrate a thresholded stress signal with gain, decay, and feedback terms.
# This helper is defined here but is not called by simulatefast_v2.
function integral_feedback2(I_prev,S,Sthresh,dt,α,k,β)
    if S-Sthresh > 0
        Sdiff = S-Sthresh
    else
        Sdiff = 0
    end
    return I_prev + dt*(α*(Sdiff) - k*I_prev + β*I_prev)
end

# Hill response with a nonnegative additive feedback term for nonzero scores.
# This helper is defined here but is not called by the current death routines.
function comphill_stress(x,pmax1,pmax2,n,k,I_curr)
    if I_curr > 0
        I = I_curr
    else
        I = 0
    end

    if x > 0
        return (pmax1)*(abs(x)^n/(k^n+abs(x)^n)) + I      #*log(stress)/(log(5)) #(stress_in+stress_curr)/stress_invmagnitude
    elseif x < 0
        return (pmax2)*(abs(x)^n/(k^n+abs(x)^n)) + I      #*log(stress)/(log(5)) #(stress_in+stress_curr)/stress_invmagnitude
    else
        return 0
    end
end

# Precompute Gaussian weights for even and odd center columns so the physical
# distances correspond to the staggered hexagonal lattice.
function build_hex_kernels(sigma, radius)
    # Precompute weights to avoid recalculating distances for every pixel
    kernel_even = Tuple{Int, Int, Float64}[]
    kernel_odd  = Tuple{Int, Int, Float64}[]

    for dj in -radius:radius
        for di in -radius:radius
            dx = dj * sqrt(3) / 2

            # The vertical distance formula shifts depending on the parity of the center column
            dy_even = di - 0.5 * mod(dj, 2)
            dy_odd  = di - 0.5 * (mod(1 + dj, 2) - 1)

            # Calculate squared Euclidean distance
            d2_even = dx^2 + dy_even^2
            d2_odd  = dx^2 + dy_odd^2

            # Gaussian weight: exp(-d^2 / (2 * sigma^2))
            push!(kernel_even, (di, dj, exp(-d2_even / (2 * sigma^2))))
            push!(kernel_odd,  (di, dj, exp(-d2_odd / (2 * sigma^2))))
        end
    end
    
    return kernel_even, kernel_odd
end
# Smooth a site-valued matrix with the parity-specific hex kernels. Normalize
# by the weights that remain in bounds at each edge of the finite lattice.
function gaussian_filter_hex(matrix; sigma=1.0, radius=ceil(Int, 3*sigma))
    nrow, ncol = size(matrix)
    
    # Allocate output matrix (Gaussian results in floats)
    smoothmat = zeros(float(eltype(matrix)), nrow, ncol)

    kernel_even, kernel_odd = build_hex_kernels(sigma, radius)

    @inbounds for j in 1:ncol
        is_even = (j % 2 == 0)
        kernel = is_even ? kernel_even : kernel_odd

        for i in 1:nrow
            val = 0.0
            weight_sum = 0.0

            for (di, dj, w) in kernel
                ni = i + di
                nj = j + dj

                # Check bounds: this inherently handles the matrix edges gracefully
                if 1 <= ni <= nrow && 1 <= nj <= ncol
                    val += matrix[ni, nj] * w
                    weight_sum += w
                end
            end
            
            # Normalize by the total weights actually applied
            smoothmat[i, j] = val / weight_sum
        end
    end

    return smoothmat
end

# Calculate local competition, smooth its absolute score over the hex lattice,
# then optionally apply a second death test from regional/local score magnitude.
# Sthresh is passed as the Gaussian width. When k_ifc > 1, the combined score
# also includes noiseconst times the unsmoothed magnitude; otherwise it uses
# only beta times the smoothed magnitude. on_switch gates the second pass.
# I_prev, alpha, and stress_thresh2 are accepted but unused in this routine.
function competition10_stress2_tracked_gaussian_localandregional_combinednormalized(matrix,death1,death2,n,k,sigma,noiserate,I_prev,Sthresh,alpha,k_ifc,noiseconst,on_switch,death_thresh,beta,stress_thresh1,stress_thresh2)
    #create indices for all cells
    cells = findall(x->x!=0,floor.(matrix))
    #create matrix for quickly finding neighbors
    neighbors=get_neighbors(matrix)
    #noise modification parameters
    #n = 3.5
    pmax1 = death1
    pmax2 = death2
    #k = 0.1

    #previous stress matrix
    oldsmatrix = get_norm_smatrix_tracked(matrix,sigma)
    #new stress matrix calculated using rate and new pull from normal distribution
    smatrix = new_smatrix_tracked_noiseratefix(matrix,oldsmatrix,sigma,noiserate)
    #convert new stress matrix into uniform distribution values (from normal values)
    newmatrix = get_uni_matrix_tracked(matrix,smatrix,sigma)

    matrixcopy = deepcopy(newmatrix)

    #matrix to store dead cell positions
    deads = []
    greendeadlocs = []

    competition_matrix = zeros(length(matrix[:,1]),length(matrix[1,:]))

    reddead = 0
    greendead = 0
    # Count cells by their own type and their number of mutant neighbors.
    comptypedict = Dict(1=>Dict(0=>0,1=>0,2=>0,3=>0,4=>0,5=>0,6=>0),2=>Dict(0=>0,1=>0,2=>0,3=>0,4=>0,5=>0,6=>0))
    comptypemat = zeros(length(matrix[:,1]),length(matrix[1,:]))

    # Scores and neighbor counts use the pre-death state for this time step.

    for i in cells
        ineighbors = neighbors[[i[1],i[2]]]
        neighborstress = []
        numred = 0
        #newneighbor = farcell_fastest([i[1],i[2]],matrix)
        for j in 1:length(ineighbors[:,1])
            if floor(matrix[ineighbors[j,1],ineighbors[j,2]]) != 0
                push!(neighborstress,smatrix[ineighbors[j,1],ineighbors[j,2]])
                if get_type(matrix[ineighbors[j,1],ineighbors[j,2]]) == 2
                    numred += 1
                end
            else
                #newneighbor = findfarthestcell_fast(matrix,[ineighbors[j,1],ineighbors[j,2]],neighbors)
                #push!(neighborstress,smatrix[newneighbor[1],newneighbor[2]])
                #if floor(matrix[newneighbor[1],newneighbor[2]]) == 2
                #    numred += 1
                #end
                push!(neighborstress,1)
            end
        end

        #if numred > 0 || floor(matrix[i])==2
        comptypedict[get_type(matrix[i])][numred] += 1
        #end
        comptypemat[i] = numred
        
        if length(neighborstress) == 0
            nothing
        else
            competition_score = smatrix[i] - mean(neighborstress)
            competition_matrix[i] = competition_score
            #if abs(competition_score) > stress_thresh
            #if competition_score > stress_thresh1 || competition_score < stress_thresh2
            #    stress_curr += 1
            #end
        end
    end
    # Regional magnitude is the Gaussian-smoothed absolute local score.
    absolute_competition_matrix = abs.(competition_matrix)
    smoothmat = gaussian_filter_hex(absolute_competition_matrix,sigma=Sthresh)
    if k_ifc > 1
        #smoothcells = findall(x->x!=0,smoothmat)
        #lowoff = maximum(smoothmat[smoothcells])/alpha
        #for i in smoothcells
        #    if smoothmat[i] < lowoff
        #        smoothmat[i] = 0
        #    end
        #end
        combined_matrix=(smoothmat.*beta .+ absolute_competition_matrix.*noiseconst)
    else
        combined_matrix = smoothmat.*beta
    end
    # This count covers all matrix sites, including empty sites with a
    # nonzero value inherited from smoothing.
    stress_curr = length(findall(x->abs(x)>stress_thresh1,combined_matrix))

    for i in findall(x->x!=0,competition_matrix)
        # First pass: the signed local competition score.
        competition_score = competition_matrix[i]
        death = comphill(competition_score,pmax1,pmax2,n,k)
        if death > death_thresh
        #if rand() < death
            matrixcopy[i] = 0
            push!(deads,i)
            numred = comptypemat[i]
            if get_type(matrix[i]) == 2
                #=if numred == 6
                    collateral += 1
                end=#
                reddead += 1
            else
                #=if numred == 0
                    collateral += 1
                end=#
                greendead += 1
                push!(greendeadlocs,i)
            end
        end
        if on_switch == true
            # Second pass: nonnegative combined magnitude. Avoid double
            # counting cells already removed by the first pass.
            competition_score2 = combined_matrix[i]
            death = comphill(competition_score2,pmax1,pmax2,n,k)
            if death > death_thresh
            #if rand() < death
                matrixcopy[i] = 0
                if i ∉ deads
                    push!(deads,i)
                    if get_type(matrix[i]) == 2
                        #=if numred == 6
                            collateral += 1
                        end=#
                        reddead += 1
                    else
                        #=if numred == 0
                            collateral += 1
                        end=#
                        greendead += 1
                        if i ∉ greendeadlocs
                            push!(greendeadlocs,i)
                        end
                    end
                end
            end
        end
    end
    # Fixed placeholder returned in the feedback position of the result tuple.
    I_curr = 1


    return matrixcopy, smatrix, competition_matrix, reddead, comptypedict, stress_curr, deads, I_curr, greendead, greendeadlocs
end

# Configuration shared by initialization, competition, stopping rules, and
# replicate runs. Despite their names, growthrate1/2 are cell-cycle lengths;
# death1/2 are Hill maxima for positive/negative scores, not for cell types.
struct parameters
    # Cell cycles and lattice/seed dimensions.
    growthrate1::Float64
    growthrate2::Float64
    r::Int
    c::Int
    startsize::Int
    startred::Int
    # Local competition response and mode (compete == 1 enables regional pass).
    death1::Float64
    death2::Float64
    n::Float64
    k::Float64
    compete::Int
    # Optional stopping conditions and delay before the regional death pass.
    endreddead::Bool
    endboundary::Bool
    memory::Int
    # Noise process and regional smoothing/combination settings. alpha is
    # carried through to the main competition routine but is unused there.
    sigma::Float64
    noiserate::Float64
    Sthresh::Float64
    alpha::Float64
    k_ifc::Float64
    noiseconst::Float64
    # Death cutoff, regional weight, stress-count cutoffs, and inheritance.
    death_thresh::Float64
    beta::Float64
    stress_thresh1::Float64
    stress_thresh2::Float64
    daughterdrift::Float64
    # Optional stop when mutants exceed half of the surviving population.
    endredmajority::Bool
end

# Pre-grow a wild-type patch, including three rounds of competition, before
# embedding it in the full lattice and replacing its center with mutant cells.
# This avoids starting the main run with a completely synchronized patch.
function initialize_tracked(p::parameters)
    #initiate matrix with size being just the start size of 
    #the wild type clone and fill entire matrix with wild type cells
    barb = init_eyeball_track(p.startsize,p.startsize,p.startsize,0,p.growthrate1,p.growthrate2)
    #run three iterations of replication and death
    for i in 1:3
        barb = replicate3_tracked(p.growthrate1,p.growthrate2,barb,p.daughterdrift)
        barb = age_cells(barb)
        bark = competition10_nostress_tracked(barb,p.death1,p.death2,p.n,p.k,p.sigma,p.noiserate,p.death_thresh,p.stress_thresh1,p.stress_thresh2)
        barb = bark[1]
    end
    #fill any holes in the resulting matrix
    numbo = 0
    while numbo < 300 && length(findall(x->x==0,barb))>0
        barb = replicate3_tracked(p.growthrate1,p.growthrate2,barb,p.daughterdrift)
        barb = age_cells(barb)
        numbo += 1
    end
    #create new full-sized matrix with old matrix (barb) at the center
    A = zeros(p.r,p.c)
    startr = round(Int, p.r/2 - p.startsize/2 + 1, RoundUp)
    startc = round(Int, p.c/2 - p.startsize/2 + 1, RoundUp)
    A[startr:startr+p.startsize-1,startc:startc+p.startsize-1]=barb
    #put mutant cells in the middle
    startinr = floor(Int,p.r/2)-floor(Int,p.startred/2)+1
    startinc = floor(Int,p.c/2)-floor(Int,p.startred/2)+1
    A[startinr:startinr+p.startred-1,startinc:startinc+p.startred-1].=2
    for i in 1:length(A[:,1])
        for j in 1:length(A[1,:])
            if A[i,j] == 2
                A[i,j] = A[i,j] + rand() + rand(1:p.growthrate2)*10
            end
        end
    end
    return A
end

# Find an occupied cell near an external empty site far from a death hole.
# The chosen cell starts the chain that shifts tissue into that hole; if the
# search finds none, choose among the most distant matrix corners instead.
function findfarthestcell(matrix,initialcoords,deadcells)
    neighbors=get_neighbors(matrix)
    line1 = copy(initialcoords[1]) #line going ^
    line3 = copy(initialcoords[1]) #line going v
    line2 = copy(initialcoords[2]) #line going >
    line4 = copy(initialcoords[2]) #line going <
    
    farthest = [[initialcoords[1],initialcoords[2]]]
    hit = 3
    number = 1
    while hit > 0 && number < length(matrix[1,:])*2
        found = []
        number+=1
        if line1+1 <= length(matrix[:,1]) #one step ^ if doesn't exceed bounds
            line1+=1
        end
        if line2+1 <= length(matrix[1,:]) #one step >
            line2+=1
        end
        if line3-1 > 0 #one step v
            line3-=1
        end
        if line4-1 > 0 #one step <
            line4-=1
        end

        #check if 0 in line1
        for i in line4:line2
            if matrix[line1,i] == 0
                if in(CartesianIndex(line1,i),deadcells) == false
                    push!(found,[line1,i])
                end
            end
        end

        #check if 0 in line3
        for i in line4:line2
            if matrix[line3,i] == 0
                if in(CartesianIndex(line3,i),deadcells) == false
                    push!(found,[line3,i])
                end
            end
        end

        #check if 0 in line2 (exclude already counted by line1 and line3)
        for i in line3+1:line1-1
            if matrix[i,line2] == 0
                if in(CartesianIndex(i,line2),deadcells) == false
                    push!(found,[i,line2])
                end
            end
        end

        #check if 0 in line 4 (exclude already counted by line1 and line3)
        for i in line3+1:line1-1
            if matrix[i,line4] == 0
                if in(CartesianIndex(i,line4),deadcells) == false
                    push!(found,[i,line4])
                end
            end
        end

        #if found a 0, do something
        if length(found) != 0
            for j in found
                jneighbors = neighbors[[j[1],j[2]]]
                for k in 1:length(jneighbors[:,1])
                    if floor(matrix[jneighbors[k,1],jneighbors[k,2]]) != 0
                        distdiff = hexdist(initialcoords,[jneighbors[k,1],jneighbors[k,2]])-hexdist(initialcoords,farthest[1])
                        if distdiff > 0
                            farthest = [[jneighbors[k,1],jneighbors[k,2]]]
                            hit = 4
                        elseif distdiff == 0
                            push!(farthest,[jneighbors[k,1],jneighbors[k,2]])
                        end
                    end
                end
            end
            hit-=1
        end
    end

    if farthest == [[initialcoords[1],initialcoords[2]]]
        farthest = [[1,1]]
        corners = [[length(matrix[:,1]),1],[1,length(matrix[1,:])],[length(matrix[:,1]),length(matrix[1,:])]]
        for i in corners
            distdiff = hexdist(initialcoords,i)-hexdist(initialcoords,farthest[1])
            if distdiff > 0
                farthest = [i]
            elseif distdiff == 0
                push!(farthest,i)
            end
        end
    end

    return sample(farthest)
end

# Close holes made by cell death by shifting a line of surviving cells toward
# each hole. bluemat marks preexisting exterior emptiness with 3 so the loop
# tracks only death holes while the shifted vacancy moves toward the edge.
# Mutates and returns matrix.
function fillholes2(matrix,deadcells)
    #create temp matrix to differentiate dead cell holes from empty epithelium space(border of growth and beyond)
    zeros = findall(x->x==0,matrix)
    #create array of non-dead cell holes, convert to cartesianindex type so it can index matrix
    outerholes = map(x->CartesianIndex(x),setdiff(zeros,deadcells))
    bluemat = deepcopy(matrix)
    bluemat[outerholes] .= 3
    #step through each dead cell hole
    numholes = length(deadcells)
    holepositions = deepcopy(deadcells)
    while numholes > 0
        hole = holepositions[1]
        #find farthest cell (named rep because this is recycled code)
        rep = findfarthestcell(matrix,hole,holepositions)
        
        #angle between x-axis and line from hole to farthest cell
        ang = atan((((rep[1]-hole[1])*sqrt(3))+sqrt(3)/2*((-1)^((hole[2])+1))*(abs(rep[2]-hole[2])%2))/((rep[2]-hole[2])*1.5))
        
        #if line between hole and rep align with one of the axes
        if ang in [-pi/2,-pi/6,pi/6,pi/2]
            if ang in [-pi/2,pi/2]
                a = rep[1]
                b = hole[1]
                foba = a > b ? 1 : -1
                for i in foba:foba:a-b
                    matrix[hole[1]+i-foba,hole[2]] = matrix[hole[1]+i,hole[2]]
                    bluemat[hole[1]+i-foba,hole[2]] = bluemat[hole[1]+i,hole[2]]
                end
                matrix[rep[1],rep[2]] = 0
                bluemat[rep[1],rep[2]] = 3
            elseif ang == pi/6
                a = rep[2]
                b = hole[2]
                foba = a > b ? 1 : -1
                for i in foba:foba:a-b
                    matrix[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = matrix[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                    bluemat[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = bluemat[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                end
                matrix[rep[1],rep[2]] = 0
                bluemat[rep[1],rep[2]] = 3
            else
                a = rep[2]
                b = hole[2]
                foba = a > b ? 1 : -1
                for i in foba:foba:a-b
                    matrix[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = matrix[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                    bluemat[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = bluemat[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                end
                matrix[rep[1],rep[2]] = 0
                bluemat[rep[1],rep[2]] = 3
            end

        #in all other cases, find the two axes that shoving will occur on
        else
            if -pi/2<ang<-pi/6
                ax1 = 1
                ax2 = 3
                direc = rand([ax1,ax2])
                if direc == ax1
                    #if ax1 == 1
                    a = rep[1] #replicating cell y-coord
                    b = hole[1] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = hole[2]
                    yintersect = Int(-floor((xintersect)/2)+rep[1]+floor((rep[2])/2))
                
                    for i in foba:foba:yintersect-b
                        matrix[hole[1]+i-foba,hole[2]] = matrix[hole[1]+i,hole[2]]
                        bluemat[hole[1]+i-foba,hole[2]] = bluemat[hole[1]+i,hole[2]]
                    end
                
                    c = rep[2] #replicating cell x-coord
                    d = hole[2] #hole x-coord
                    foba2 = c > d ? 1 : -1
                
                    #yax = floor((xax+1)/2)+ya-floor((xa+1)/2)
                    for j in foba2:foba2:c-d
                        matrix[Int(-floor((hole[2]+j-foba2)/2)+rep[1]+floor((rep[2])/2)),hole[2]+j-foba2] = matrix[Int(-floor((hole[2]+j)/2)+rep[1]+floor((rep[2])/2)),hole[2]+j]
                        bluemat[Int(-floor((hole[2]+j-foba2)/2)+rep[1]+floor((rep[2])/2)),hole[2]+j-foba2] = bluemat[Int(-floor((hole[2]+j)/2)+rep[1]+floor((rep[2])/2)),hole[2]+j]
                    end
                    #replicating cell
                    matrix[rep[1],rep[2]] = 0
                    bluemat[rep[1],rep[2]] = 3
                
                else
                    #if ax1 == 1
                    a = rep[2] #replicating cell x-coord
                    b = hole[2] #hole x-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = rep[2]
                    yintersect = Int(-floor((xintersect)/2)+hole[1]+floor((hole[2])/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = matrix[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                        bluemat[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = bluemat[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                    end
                
                    c = rep[1] #replicating cell y-coord
                    d = yintersect #hole y-coord
                    foba2 = c > d ? 1 : -1
                
                    for j in foba2:foba2:c-d
                        matrix[yintersect+j-foba2,xintersect] = matrix[yintersect+j,xintersect]
                        bluemat[yintersect+j-foba2,xintersect] = bluemat[yintersect+j,xintersect]
                    end
                    #replicating cell
                    matrix[rep[1],rep[2]] = 0
                    bluemat[rep[1],rep[2]] = 3
                end
            elseif -pi/6<ang<pi/6
                ax1 = 2
                ax2 = 3
                #check to see if shoving could reach above or below dimensions of matrix (in which case, switches to start with other axis)
                ax1intersect = Int(floor((Int(floor((hole[2]+1)/2)+floor((rep[2])/2)+rep[1]-hole[1])+1)/2)+hole[1]-floor((hole[2]+1)/2))
                ax2intersect = Int(floor((Int(floor((rep[2]+1)/2)+floor((hole[2])/2)+hole[1]-rep[1])+1)/2)+rep[1]-floor((rep[2]+1)/2))
                if ax1intersect > length(matrix[:,1]) || ax1intersect < 1
                    direc = ax2
                elseif ax2intersect > length(matrix[:,1]) || ax2intersect < 1
                    direc = ax1
                else
                    direc = rand([ax1,ax2])
                end
                if direc == ax1
                    #if ax1 == 1
                    a = rep[2] #replicating cell y-coord
                    b = hole[2] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = Int(floor((hole[2]+1)/2)+floor((rep[2])/2)+rep[1]-hole[1])
                    yintersect = Int(floor((xintersect+1)/2)+hole[1]-floor((hole[2]+1)/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = matrix[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                        bluemat[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = bluemat[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                    end
                
                    for j in foba:foba:a-xintersect
                        matrix[Int(-floor((xintersect+j-foba)/2)+rep[1]+floor((rep[2])/2)),xintersect+j-foba] = matrix[Int(-floor((xintersect+j)/2)+rep[1]+floor((rep[2])/2)),xintersect+j]
                        bluemat[Int(-floor((xintersect+j-foba)/2)+rep[1]+floor((rep[2])/2)),xintersect+j-foba] = bluemat[Int(-floor((xintersect+j)/2)+rep[1]+floor((rep[2])/2)),xintersect+j]
                    end
                    #replicating cell
                    matrix[rep[1],rep[2]] = 0
                    bluemat[rep[1],rep[2]] = 3
                
                else
                    #if ax1 == 1
                    a = rep[2] #replicating cell y-coord
                    b = hole[2] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = Int(floor((rep[2]+1)/2)+floor((hole[2])/2)+hole[1]-rep[1])
                    yintersect = Int(floor((xintersect+1)/2)+rep[1]-floor((rep[2]+1)/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = matrix[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                        bluemat[Int(-floor((hole[2]+i-foba)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i-foba] = bluemat[Int(-floor((hole[2]+i)/2)+hole[1]+floor((hole[2])/2)),hole[2]+i]
                    end
                
                    for j in foba:foba:a-xintersect
                        matrix[Int(floor((xintersect+j-foba+1)/2)+rep[1]-floor((rep[2]+1)/2)),xintersect+j-foba] = matrix[Int(floor((xintersect+j+1)/2)+rep[1]-floor((rep[2]+1)/2)),xintersect+j]
                        bluemat[Int(floor((xintersect+j-foba+1)/2)+rep[1]-floor((rep[2]+1)/2)),xintersect+j-foba] = bluemat[Int(floor((xintersect+j+1)/2)+rep[1]-floor((rep[2]+1)/2)),xintersect+j]
                    end
                    #replicating cell
                    matrix[rep[1],rep[2]] = 0
                    bluemat[rep[1],rep[2]] = 3
                end
            elseif pi/6<ang<pi/2
                ax1 = 1
                ax2 = 2
                direc = rand([ax1,ax2])
                if direc == ax1
                    #if ax1 == 1
                    a = rep[1] #replicating cell y-coord
                    b = hole[1] #hole y-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = hole[2]
                    yintersect = Int(floor((xintersect+1)/2)+rep[1]-floor((rep[2]+1)/2))
                
                    for i in foba:foba:yintersect-b
                        matrix[hole[1]+i-foba,hole[2]] = matrix[hole[1]+i,hole[2]]
                        bluemat[hole[1]+i-foba,hole[2]] = bluemat[hole[1]+i,hole[2]]
                    end
                
                    c = rep[2] #replicating cell x-coord
                    d = hole[2] #hole x-coord
                    foba2 = c > d ? 1 : -1
                
                    #yax = floor((xax+1)/2)+ya-floor((xa+1)/2)
                    for j in foba2:foba2:c-d
                        matrix[Int(floor((hole[2]+j-foba2+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+j-foba2] = matrix[Int(floor((hole[2]+j+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+j]
                        bluemat[Int(floor((hole[2]+j-foba2+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+j-foba2] = bluemat[Int(floor((hole[2]+j+1)/2)+rep[1]-floor((rep[2]+1)/2)),hole[2]+j]
                    end
                    #replicating cell
                    matrix[rep[1],rep[2]] = 0
                    bluemat[rep[1],rep[2]] = 3
                else
                    #if ax1 == 1
                    a = rep[2] #replicating cell x-coord
                    b = hole[2] #hole x-coord
                    foba = a > b ? 1 : -1 #forward or backward
                
                    #find intersection of two axes, to know when to stop for loop (how far up before turning to go diagonal)
                    xintersect = rep[2]
                    yintersect = Int(floor((xintersect+1)/2)+hole[1]-floor((hole[2]+1)/2))
                
                    for i in foba:foba:xintersect-b
                        matrix[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = matrix[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                        bluemat[Int(floor((hole[2]+i-foba+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i-foba] = bluemat[Int(floor((hole[2]+i+1)/2)+hole[1]-floor((hole[2]+1)/2)),hole[2]+i]
                    end
                
                    c = rep[1] #replicating cell y-coord
                    d = yintersect #hole y-coord
                    foba2 = c > d ? 1 : -1
                
                    yax = floor((hole[2]+yintersect-b+1)/2)+hole[1]-floor((hole[2]+1)/2)
                    for j in foba2:foba2:c-d
                        matrix[yintersect+j-foba2,xintersect] = matrix[yintersect+j,xintersect]
                        bluemat[yintersect+j-foba2,xintersect] = bluemat[yintersect+j,xintersect]
                    end
                    #replicating cell
                    matrix[rep[1],rep[2]] = 0
                    bluemat[rep[1],rep[2]] = 3
                end
            end
        end
        #reset holepositions for while loop
        holepositions = findall(x->x==0,bluemat)
        numholes = length(holepositions)
    end
    return matrix
end

# True when at least three of four Boolean inputs are true. Currently unused.
function atLeastThree(a,b,c,d)
    return ((a && b && c)) || ((a && b && d)) || ((a && c && d)) || ((b && c && d))
end

# For each wild-type death, measure the shortest hex distance to a mutant.
# When no mutants remain, the fallback constructs a reference line using
# dim(matrix), which must be available in the calling environment.
function find_deathdistances(matrix,greendeadlocs)
    mutants = findall(x->x==2,get_types(matrix))
    if length(mutants) == 0
        mid = dim(matrix) ÷ 2
        mutants = [[i, mid] for i in 1:dim(matrix)]
    end
    shortest_distances = []
    for i in greendeadlocs
        distances = []
        for j in mutants
            distance = hexdist([i[1],i[2]],[j[1],j[2]])
            push!(distances,distance)
        end
        push!(shortest_distances,minimum(distances))
    end
    return shortest_distances
end

# Run one simulation and collect per-step mutant counts, death counts, total
# occupancy, and wild-type death distances. Stops at timesteps or at one of
# the extinction, boundary, or majority conditions below.
function simulatefast_v2(timesteps,p::parameters)

    matrix = initialize_tracked(p)
    anum = [] #stores num reds

    reddeads = []
    greendeads = []
    totcells = []
    I_prev = 0
    deathdists = []

    nbs = get_neighbors(matrix)
    ag = get_adjacencymatrix(matrix,p.r,p.c,nbs)


    for i in 1:timesteps
        # Division and displacement occur before aging and competition.
        matrix=replicate3_tracked_pathfinder_faster_redshoverule_movebubbles_edgefinder(p.growthrate1,p.growthrate2,matrix,p.daughterdrift,p.r,p.c,ag)
        matrix = age_cells(matrix)
        #competition
        if p.compete == 1
            # Delay the regional death pass for the configured number of steps.
            if i > p.memory
                on_switch = true
            else
                on_switch = false
            end
            matrix,smat,compmat,reddead,compdict,stress_curr,deadcells,I_prev,greendead,greendeadlocs = competition10_stress2_tracked_gaussian_localandregional_combinednormalized(matrix,p.death1,p.death2,p.n,p.k,p.sigma,p.noiserate,I_prev,p.Sthresh,p.alpha,p.k_ifc,p.noiseconst,on_switch,p.death_thresh,p.beta,p.stress_thresh1,p.stress_thresh2)
            deathdistances = find_deathdistances(matrix,greendeadlocs)
            push!(deathdists,deathdistances)
        else
            matrix,smat,compmat,reddead,compdict,stress_curr,deadcells,greendead = competition10_nostress_tracked(matrix,p.death1,p.death2,p.n,p.k,p.sigma,p.noiserate,p.death_thresh,p.stress_thresh1,p.stress_thresh2)
        end
        # Relocate survivors into death holes before recording cell counts.
        matrix=fillholes2(matrix,deadcells)
        #store matrices
        push!(anum,length(findall(x->x==2,get_types(matrix))))
        push!(reddeads,reddead)
        push!(greendeads,greendead)
        totcellcount = count(!iszero,matrix)
        push!(totcells,totcellcount)

        # Stop when all cells die or all survivors are mutants. The remaining
        # conditions can be enabled independently in the parameter set.
        if totcellcount == 0
            break
        elseif totcellcount == anum[end]
            break
        end
        if p.endreddead
            if anum[end] == 0
                break
            end
        end
        if p.endboundary
            #if atLeastThree(matrix[1,1] != 0, matrix[1,end] != 0, matrix[end,1] != 0, matrix[end,end] != 0)
            #    break
            #end
            # This boundary test assumes the lattice has row and column 100.
            if sum(matrix[:,1])+sum(matrix[:,100])+sum(matrix[1,:])+sum(matrix[100,:]) != 0
                break
            end
        end
        if p.endredmajority
            if anum[end]/totcellcount > 0.5
                break
            end
        end
    end

    return anum,greendeads,totcells,reddeads,deathdists
end

# Repeat independent simulations and return five collections: mutant counts,
# wild-type deaths per wild-type cell, total cells, mutant deaths per mutant,
# and distances from wild-type deaths to mutants. Ratios may be undefined
# when their corresponding cell count reaches zero.
function parameter_test_full_fast(sims,timesteps,p::parameters)
    resultbox = []
    for element in 1:5
        push!(resultbox,[])
    end
    for element in eachindex(resultbox)
        for j in 1:sims
            push!(resultbox[element],[])
        end
    end

    for i in 1:sims
        num,greendeads,totcells,reddeads,deathdists = simulatefast_v2(timesteps,p);
        resultbox[1][i] = num
        resultbox[2][i] = greendeads./(totcells.-num)
        resultbox[3][i] = totcells
        resultbox[4][i] = reddeads./num
        resultbox[5][i] = deathdists
        #print("$i ")
    end
    return resultbox
end

# Run 50 treatment simulations and 5 controls without a mutant seed.
# Store both results in the caller-provided global dictionaries
# result_collect and control_result_collect under consecutive "rN" keys.
function runparamset(p::parameters)
    control_params = parameters(p.growthrate1,p.growthrate2,p.r,p.c,p.startsize,0,p.death1,p.death2,p.n,p.k,p.compete,false,p.endboundary,p.memory,p.sigma,p.noiserate,p.Sthresh,p.alpha,p.k_ifc,p.noiseconst,p.death_thresh,p.beta,p.stress_thresh1,p.stress_thresh2,p.daughterdrift,p.endredmajority)

    currlenr = length(result_collect)+1
    #result,example = parameter_test_save_example(100,300,p);
    result = parameter_test_full_fast(50,200,p);
    
    currlenrc = length(control_result_collect)+1
    control_result = parameter_test_full_fast(5,200,control_params);

    result_collect["r$currlenr"] = result
    control_result_collect["r$currlenrc"] = control_result
    #example_collect["r$currlenr"] = example
end
