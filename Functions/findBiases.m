function biases = findBiases(cps, weights, featureVector, symbols, epsilon, cls)
    biases = zeros(size(cps,1),1);
    
    % Find bias for each critical point
    for i = 1:size(cps,1)
        % Calculate the bias term such that weights'*cp+bias = 0
        cp = subs(featureVector, symbols, [cps(i,:)]);
        bias = -cp * weights;
        bias = bias + epsilon * sign(bias-cls.Bias);
        biases(i) = bias;
    end
end

