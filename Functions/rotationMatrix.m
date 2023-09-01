function m = rotationMatrix(symbols, theta, dim1, dim2)

    % create identity matrix
    m = zeros([size(symbols, 1) size(symbols, 1)]);
    for i = 1:size(symbols)
        m(i,i) = 1;
    end
    
    % Set rotate among dimensions dim1 and dim2
    m(dim1, dim1) = cosd(theta);
    m(dim2, dim2) = cosd(theta);
    m(dim1, dim2) = -sind(theta);
    m(dim2, dim1) = sind(theta);
end