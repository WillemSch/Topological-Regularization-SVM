function nodeIndex = getNodeIndex(coordinates, m)
    % Convert coordinates in the n-dimensional grid to node index
    nodeIndex = 1;
    for i = 1:length(coordinates)
        nodeIndex = nodeIndex + (coordinates(i)-1) * (m^(length(coordinates)-i));
    end
end