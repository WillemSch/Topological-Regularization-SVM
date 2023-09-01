function numComponents = countConnectedComponents(g)
    numComponents = 0;
    while size(g.Nodes,1) > 0
        numComponents = numComponents + 1;
        v = dfsearch(g,1);
        g = rmnode(g, v);
    end
end