function scores = predictMan(bias, weights, X, featureVector, symbols)
    
    % Transform all data to be predicted to the feature space
    XFeatures = zeros(size(X,1), size(featureVector,2));
    for i = 1:size(X,1)
        XFeatures(i,:) = subs(featureVector, symbols, [X(i,:)]);
    end
    
    % Initialize score array
    scores = zeros(size(XFeatures,1),1);

    % Predict all points in X
    for i = 1:size(XFeatures,1)
        p = dot(weights, XFeatures(i,:)) + bias;
        scores(i) = p;
    end
end

