function X = randomPoints(limits, n)
    d = size(limits,1);
    X = rand(n, d);

    for i = 1:d
        X(:,i) = (limits(i,2)-limits(i,1)) .* X(:,i) + limits(i,1);
    end
end

