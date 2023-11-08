function scores = predictMan(bias, weights, X, featureVector, symbols)
    
    featureVector = simplify(featureVector);
    % Transform all data to be predicted to the feature space
    XFeatures = zeros(size(X,1), size(featureVector,2));
    for i = 1:size(X,1)
        XFeatures(i,:) = subs(featureVector, symbols, [X(i,:)]);
    end

    % Initialize score array
    scores = zeros(size(XFeatures,1),1);

    repw = repmat(weights,1,size(XFeatures,1));
    scores = (dot(repw,XFeatures') + bias)';
end

