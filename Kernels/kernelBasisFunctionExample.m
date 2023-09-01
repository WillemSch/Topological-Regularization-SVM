function K_matrix = kernelBasisFunctionExample(X, Y)
    % Kernel which results in mapping of (x) -> (x, x^4-x^2+0.1x)
    nX = size(X, 1);
    nY = size(Y, 1);
    
    % Intialize kernel matrix
    K_matrix = zeros(nX, nY);
    
    % Loop through all datapoints in X and Y
    for i = 1:nX
        for j = 1:nY
            x = X(i, :);
            y = Y(j, :);
            % calculate the value of the kernel matrix
            K_matrix(i, j) = x * y' + (x(1)^4 -x(1)^2+0.1*x(1)) * (y(1)^4 -y(1)^2+0.1*y(1));
        end
    end
end