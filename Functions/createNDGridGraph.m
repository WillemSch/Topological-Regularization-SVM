function adjacencyMatrix = createNDGridGraph(n, m)
    % n: Number of dimensions
    % m: Number of nodes along each dimension
    
    % Total number of nodes in the n-dimensional grid
    totalNodes = m^n;
    
    % Initialize adjacency matrix
    adjacencyMatrix = zeros(totalNodes);
    
    % Generate coordinates for each node
    coordinates = cell(1, n);
    for i = 1:n
        coordinates{i} = 1:m;
    end
    
    % Generate edges for the n-dimensional grid
    for node = 1:totalNodes
        nodeCoordinates = getCoordinates(node, m, n);
        
        % Generate edges for each dimension
        for dim = 1:n
            % Generate neighbors in positive direction
            if nodeCoordinates(dim) < m
                neighborCoordinates = nodeCoordinates;
                neighborCoordinates(dim) = neighborCoordinates(dim) + 1;
                neighbor = getNodeIndex(neighborCoordinates, m);
                adjacencyMatrix(node, neighbor) = 1;
            end
            
            % Generate neighbors in negative direction
            if nodeCoordinates(dim) > 1
                neighborCoordinates = nodeCoordinates;
                neighborCoordinates(dim) = neighborCoordinates(dim) - 1;
                neighbor = getNodeIndex(neighborCoordinates, m);
                adjacencyMatrix(node, neighbor) = 1;
            end
        end
    end
end

function coordinates = getCoordinates(nodeIndex, m, n)
    % Convert node index to coordinates in the n-dimensional grid
    coordinates = zeros(1, n);
    for i = n:-1:1
        coordinates(i) = mod(nodeIndex-1, m) + 1;
        nodeIndex = floor((nodeIndex-1)/m) + 1;
    end
end