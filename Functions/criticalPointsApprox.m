function cps = criticalPointsApprox(symbols, high_dim_vector, weights, limits, attempts)
    
    % Initialize symbolic array to store all equations
    eqs = sym(size(symbols,2));
    
    high_dim_vector = simplify(high_dim_vector);

    % Find all partial derivatives and store them equal to 0 in eqs
    for i = 1:size(symbols,2)
        dxi = diff(high_dim_vector, symbols(i));
        eqs(i) = dot(dxi', weights) == 0;
    end

    % Find the solutions that hold for all equations
    cps = zeros(0, size(symbols,2));
    for i = 1:attempts
        sol = vpasolve(eqs,symbols, limits,"Random",true);

        cp = zeros(1,size(symbols,2));
        for j = 1:size(symbols,2)
            cp(j) = (sol.(char(symbols(j))));
        end

        if ~ismember(cp, cps)
            cps(end+1,:) = cp;
        end
    end
end 