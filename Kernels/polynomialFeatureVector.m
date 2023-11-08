function phi = RBFApproxFeatureVector(Q, x, gamma)    
    % x is the input feature vector
    % Q is the number of dimensions to consider
    % gamma is the gamma value of the RBF kernel
    
    n = numel(x);  % Number of input features
    phi = sym(zeros((d + 1) * n, 1));  % Initialize the feature mapping vector
    
    % Construct the feature mapping
    index = 1;
    for i = 0:Q
        for j = 1:n
            phi(index) = sqrt(nchoosek(d, i)) * (x(j) ^ i);
            index = index + 1;
        end
    end
end

