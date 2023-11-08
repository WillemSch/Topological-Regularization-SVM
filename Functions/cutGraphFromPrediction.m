function h = cutGraphFromPrediction(g, labels)
    h = g;
    for i = 1:size(g.Edges.EndNodes,1)
        v = g.Edges.EndNodes(i,1);
        t = g.Edges.EndNodes(i,2);
        if labels(v) ~= labels(t)
            h = rmedge(h, v, t);
        end
    end
end

