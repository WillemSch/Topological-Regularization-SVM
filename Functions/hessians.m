function h = hessians(f, symbols)
    % Initialize hessian matrix
    h = {};

    for i = 1:size(symbols, 2)-1
        for j = 1:size(symbols, 2)-1
            % Find all second order derivatives
            ddf = diff(diff(f, symbols{i}), symbols{j});
            h{i,j} = ddf;
        end
    end
end