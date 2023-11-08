addpath("./")                       % Import functions
rng(1);                             % For reproducibility
syms x1;                            % Create symbol set (must match nr# input features of data)

[X,Y] = createDataset();

% Ensure hard margin constraint
cls = fitcsvm(X,Y,'KernelFunction', 'kernelBasisFunctionExample','BoxConstraint',Inf);

% Find the original margin
m = max(margin(cls, X(cls.IsSupportVector, :), Y(cls.IsSupportVector)));

plot_svms_results_with_model(X,Y,cls,[x1], m)


function plot_svms_results_with_model(X, y, cls, symbols, m)
    syms x1;
    high_dim_vector = [x1, x1^4-x1^2+0.1*x1];

    % Find critical points
    weights = normalVector(cls, high_dim_vector, symbols, 1);
    cps = criticalPoints(symbols, high_dim_vector, weights);
    biases = findBiases(cps, weights, high_dim_vector, symbols, 2^-10, cls);
    
    % Create Test Dataset
    [XTest, YTest] = createTestDataset(100);

    for bias = [cls.Bias, biases']
        disp("Bias=" + bias)
        % Change in bias (distance the separator moved)
        d = bias-cls.Bias;

        % Sample points for using a grid for a decision boundary plot
        sampleResolution = 1000;
        x_vals = linspace(-1.75, 1.5, sampleResolution);
        g = nDGridGraph(size(symbols,1), sampleResolution);
        scores = predictMan(bias, weights, x_vals', high_dim_vector, symbols);
        y_vals = sign(scores);

        % Remove edges between nodes with different labels
        g = cutGraphFromPrediction(g, y_vals);

        % Count connected components in graph
        numComponents = countConnectedComponents(g);

        % Calculate Accuracy on test dataset
        acc = accuracy(YTest, sign(predictMan(bias, weights, XTest, high_dim_vector, symbols)));
        
        % Start figure
        figure;
        hold on;

        % Plot data points
        XFeature = X.^4-X.^2+0.1*X;
        scatter(X(y == 1),  XFeature(y == 1), 'b', 'o', 'filled');
        scatter(X(y == -1), XFeature(y == -1), 'r', 'x');
        
        % Plot support vectors
        scatter(X(cls.IsSupportVector, :), XFeature(cls.IsSupportVector), 75, 'ko');
        
        % Predict decision boundary values using the SVM model    
        plot(x_vals, zeros(size(x_vals,2)) - (bias/norm(weights)), "LineWidth", 1, 'Color', 'k');
        
        % Plot function
        scatter(x_vals(y_vals == 1),  x_vals(y_vals == 1).^4-x_vals(y_vals == 1).^2+0.1*x_vals(y_vals == 1),   1,'b');
        scatter(x_vals(y_vals == -1), x_vals(y_vals == -1).^4-x_vals(y_vals == -1).^2+0.1*x_vals(y_vals == -1),1,'r');
        
        xlim([min(X) - 0.5, max(X) + 0.5]);
        ylim([-1, 1]);
        
        legend('Class 1', 'Class -1', 'Support Vectors', 'Decision Boundary');
        if d == 0
            title("Baseline, β_0=" + numComponents + ", accuracy=" + acc);
        else
            title("PTLS, D_m="+ (abs(d)/m) + ", β_0=" + numComponents + ", accuracy=" + acc);
        end
        xlabel('Input dimension');
        ylabel('Feature dimension');
        hold off;
    end

    for cp = cps'
        % Set the class of the critical point to the opposite of the
        % predicted class
        cpClass = -predict(cls, cp);
        disp("Refit with addition of x=" + cp + " with class="+cpClass)
        % Add the critical point to the dataset
        [X2, Y2] = createDataset();
        X2(end+1) = cp;
        Y2(end+1) = cpClass;

        % Refit the SVM
        newCls = fitcsvm(X2,Y2,'KernelFunction','kernelBasisFunctionExample','BoxConstraint',Inf,'Prior','uniform');
        weights = normalVector(newCls, high_dim_vector, symbols, 1);

        % Sample points for smoother decision boundary plot
        sampleResolution = 1000;
        x_vals = linspace(-1.75, 1.5, sampleResolution);
        g = nDGridGraph(size(symbols,1), sampleResolution);
        scores = predict(newCls, x_vals');
        y_vals = sign(scores);

        % Remove edges between nodes with different labels
        g = cutGraphFromPrediction(g, y_vals);

        % Count connected components in graph
        numComponents = countConnectedComponents(g);
        
        % Calculate Accuracy on test dataset
        acc = accuracy(YTest, sign(predict(newCls, XTest)));

        % Start figure
        figure;
        hold on;

        % Plot data points
        XFeature = X2.^4-X2.^2+0.1*X2;
        scatter(X2(Y2 == 1),  XFeature(Y2 == 1), 'b', 'o', 'filled');
        scatter(X2(Y2 == -1), XFeature(Y2 == -1), 'r', 'x');
        
        % Plot support vectors
        scatter(X2(newCls.IsSupportVector, :), XFeature(newCls.IsSupportVector), 75, 'ko');
        
        % Predict decision boundary values using the SVM model
        c = newCls.Bias;
        x1 = linspace(-1.75, 1.5, 2);
        x2 = -(weights(1)/weights(2)*x1)-c/weights(2);
        % Plot the line
        plot(x1, x2, 'b', 'LineWidth', 1, 'Color', 'k');
        
        % Plot function
        scatter(x_vals(y_vals == 1),  x_vals(y_vals ==  1).^4-x_vals(y_vals ==  1).^2+0.1*x_vals(y_vals ==  1),1,'b');
        scatter(x_vals(y_vals == -1), x_vals(y_vals == -1).^4-x_vals(y_vals == -1).^2+0.1*x_vals(y_vals == -1),1,'r');
        
        xlim([min(X2) - 0.5, max(X2) + 0.5]);
        ylim([-1, 1]);
        
        legend('Class 1', 'Class -1', 'Support Vectors', 'Decision Boundary');
        title("AACP, Critical Point: x=" + round(cp,3) + ", β_0=" + numComponents + ", accuracy=" + acc);
        xlabel('Input dimension');
        ylabel('Feature dimension');
        disp(acc);
        hold off;
    end
end

function [X,Y] = createDataset()
    X = [-1.2,-1.1,-1.0466805318046,-0.9376362514803,-0.9,-0.88,-0.8,-0.76,-0.66,-0.5,-0.4398137946008,0,0.95,1]';
    Y = [1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1]';
end

function [XTest,YTest] = createTestDataset(testSize)    
    XTest = zeros(testSize,1);
    YTest = ones(testSize,1);
    for i = 1:testSize
	    if i < testSize / 2
            if mod(i,3) == 0
		        XTest(i) = (-.5).*rand(1) - 1;
            else
		        XTest(i) = (1.7).*rand(1) - .2;
            end
	    else
		    YTest(i) = -1;
		    XTest(i) = (-.4).*rand(1) - .5;
	    end
    end
end