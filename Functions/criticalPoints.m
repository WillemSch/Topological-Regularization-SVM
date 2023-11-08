function cps = criticalPoints(symbols, high_dim_vector, weights)
    
    % Initialize symbolic array to store all equations
    eqs = sym(size(symbols,2));
    
    high_dim_vector = simplify(high_dim_vector);

    % Find all partial derivatives and store them equal to 0 in eqs
    for i = 1:size(symbols,2)
        dxi = diff(high_dim_vector, symbols(i));
        eqs(i) = dot(dxi', weights) == 0;
    end

    % Find the solutions that hold for all equations
    sols = solve(eqs,symbols);
    
    % Return of solve is different when there is 1 variable vs multiple
    if size(symbols, 2) == 1
        cpsSize = size(sols,1);
    else
        cpsSize = size(sols.x1,1);
    end

    % Transform solutions to coordinate tuples
    cps = zeros(cpsSize, size(symbols,2));
    for i = 1:cpsSize
        for j = 1:size(symbols,2)
            if size(symbols, 2) == 1
                cps(i,j) = sols(i);
            else
                s = symbols(j);
                cps(i,j) = sols.(char(s))(i);
            end
        end
    end
end 