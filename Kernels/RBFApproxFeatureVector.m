function phi = RBFApproxFeatureVector(Q, x, gamma)    
    % x is the input feature vector
    % Q is the number of dimensions to consider
    
    Q = Q+1; % as 0 <= n_i <= Q, there are Q+1 possible values
    D = numel(x);
    phi = sym(zeros(Q^D,1));  % Initialize the feature mapping vector

    ns = multiIndices(Q, D);

    a = exp(-gamma*(norm(x)^2));
    
    % Construct the feature mapping
    for i = 1:size(ns,2)
        n = ns(:,i);
        phi(i) = b(n, gamma)*c(x,n);
    end
    phi = a * phi;
end

function ns = multiIndices(Q, D)
    % Q is the number of dimensions to consider
    % D is the number of input space dimensions

    ns = zeros(D,Q^D);
    for i = 1:Q^D
        for d = 1:D
            ns(d,i) = mod(floor(i/(Q^(d-1))), Q);
        end
    end
end

function x = b(n, gamma)
    % n is the multi-index for which to calculate b
    
    numerator = 1;
    denominator = 1;
    for i = numel(n)
        ni = n(i);
        numerator = numerator * (2*gamma)^ni;
        denominator = denominator * factorial(ni);
    end
    x = sqrt(numerator/denominator);
end

function p = c(x, n)
    % x is the datapoint for which to calculate c
    % n is the multi-index for which to calculate c
    
    p = 1;
    for i = 1:numel(x)
        p = p * x(i)^n(i);
    end
end