function K_matrix = kernelNoiseExample(X, Y)
    % Kernel which results in mapping of (x) -> (x, x^3+x^2)
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
            K_matrix(i, j) = x * y' + (x(1)^3 +x(1)^2) * (y(1)^3 +y(1)^2);
        end
    end
end