function scores = predictManNoTransform(bias, weights, X)

    % Initialize score array
    scores = zeros(size(X,1),1);

    repw = repmat(weights,1,size(X,1));
    scores = (dot(repw,X') + bias)';
end
