syms x1 x2 real
rng(2)

% Create Training Dataset
[X,Y] = createDataset(100);
% Create Test Dataset
[XTest, YTest] = createDataset(200);

% Model hyperparameters
kernelScale = 1;
Q = 15;

% Create the mapping function
phi = RBFApproxFeatureVector(Q, [x1, x2], 4^2);

% Transform data to feature space
XFeature = [];
for i = 1:size(X,1)
    XFeature(i,:) = subs(phi, [x1, x2], X(i,:));
end

XTestFeature = [];
for i = 1:size(XTest,1)
    XTestFeature(i,:) = subs(phi, [x1, x2], XTest(i,:));
end

disp("Fitting Baseline SVM")
cls = fitcsvm(XFeature, Y, 'KernelFunction', 'linear', 'BoxConstraint', Inf);
[pred, scores] = predict(cls, XFeature);

disp("Finding normal vector")
weights = normalVector(cls, phi, [x1, x2], 1);


disp("Locating critical points")
cps = criticalPointsApprox([x1 x2], phi, weights,[-0.2 1;-0.2 1],100);

disp("Calculating Biases")
biases = findBiases(cps,weights,phi',[x1 x2],2^-10,cls);


%Sample points for using a grid for a decision boundary plot
disp("Creating Grid")
h1 = (max(X(:,1))+.1 - min(X(:,1)))/60; % Mesh grid step size
h2 = (max(X(:,2))+.1 - min(X(:,2)))/60; % Mesh grid step size
[X1,X2] = meshgrid(min(X(:,1)):h1:max(X(:,1))+.1,...
    min(X(:,2)):h2:max(X(:,2))+.1);

x_vals = [];
xs = [X1(:),X2(:)];
disp("Mapping grid to feature space")
for i = 1:size(xs,1)
    x_vals(i,:) = subs(phi, [x1, x2], xs(i,:));
end



% Sample points using a delaunay graph
disp("Processing Delaunay graph")
x_delaunay = randomPoints([[max(X(:,1))+.1, min(X(:,1))];[max(X(:,2))+.1, min(X(:,2))]], 1000);
triangulation = delaunay(x_delaunay);

% Get the edges from the Delaunay triangulation
edges = [];
[numTriangles, ~] = size(triangulation);
for j = 1:numTriangles
    triangle = triangulation(j, :);
    % Generate three edges for each triangle
    edges = [edges; [triangle(1), triangle(2)]; [triangle(2), triangle(3)]; [triangle(3), triangle(1)]];
end

disp("Mapping Delaunay graph to feature space")
x_delaunay_feature = [];
for j = 1:size(x_delaunay,1)
    x_delaunay_feature(j,:) = subs(phi, [x1, x2], x_delaunay(j,:));
end

m = max(margin(cls, XFeature(cls.IsSupportVector, :), Y(cls.IsSupportVector)));

for i = 1:size(cps,1)+1
    if i > size(cps,1)
        bias = cls.Bias;
    else
        bias = biases(i);
        c = cps(i,:);
    end

    disp("Bias=" + bias)
    % Change in bias (distance the separator moved)
    d = bias-cls.Bias;
    
    % Create grid graph
    disp("  Processing grid graph")
    sampleResolution = (size(X1,1));
    g = nDGridGraph(2, sampleResolution);
    scores = predictManNoTransform(bias, weights, x_vals);
    y_vals = sign(scores);
    scoreGrid = reshape(y_vals(:),size(X1,1),size(X2,2));

    % Remove edges between nodes with different labels
    g = cutGraphFromPrediction(g, y_vals);

    
    % Create a graph object from the extracted edges
    g_del = graph(edges(:, 1), edges(:, 2));
    
    scores = predictManNoTransform(bias, weights, x_delaunay_feature);
    y_vals = sign(scores);

    % Remove edges between nodes with different labels
    g_del = cutGraphFromPrediction(g_del, y_vals);
    
    disp("  Calculating performance measures")
    % Count connected components in graph
    numComponents = countConnectedComponents(g);
    numComponents_del = countConnectedComponents(g_del);

    % Calculate Accuracy on test dataset
    acc = accuracy(YTest, sign(predictManNoTransform(bias, weights, XTestFeature)));

    % Create figure
    figure
    hold on
    contourf(X1,X2,scoreGrid,1)

    map = [255 115 115
        255 115 115
        131 115 255
        131 115 255] / 255;
    colormap(map)
    
    scatter(X(Y == 1,1),  X(Y == 1,2), 'b', 'o', 'filled');
    scatter(X(Y == -1,1), X(Y == -1,2), 'r', 'x');
    scatter(cps(:,1), cps(:,2), 'black', 'pentagram', 'filled')
    
    svInd = cls.IsSupportVector;
    plot(X(svInd,1),X(svInd,2),'o','MarkerSize',10)
    colorbar;

    title("PTLS, D_m="+ (abs(d)/m) + ", β_0=" + numComponents + ", accuracy=" + acc);
    % Print LaTeX style table:
    % disp("Nr. " + i + ", $("+c(1)+","+c(2)+")$ & $"+round((abs(d)/m),4)+"$ & $"+numComponents+"$ & $"+numComponents_del+"$ & $"+acc+"$\%\\");

    xlabel("x")
    ylabel("y")

    legend('Prediction', 'Class 1', 'Class -1', 'Critical Points', 'Support Vectors')
    hold off
end


for cp = cps'
    % Set the class of the critical point to the opposite of the
    % predicted class'
    cpFeature = double(subs(phi', [x1 x2]', cp));
    cpClass = -predict(cls, cpFeature);
    disp("Refit with addition of x=" + cp(1) + ", y=" + cp(2) + " with class="+cpClass)

    Xcp = X;
    XFeaturecp = XFeature;
    Ycp = Y;

    Xcp(end+1,:) = cp;
    XFeaturecp(end+1,:) = cpFeature;
    Ycp(end+1) = cpClass;

    % Refit the SVM
    disp("  Fitting new SVM")
    newCls = fitcsvm(XFeaturecp,Ycp,'KernelFunction','linear','BoxConstraint',Inf,'Prior','uniform');
    weights = normalVector(newCls, phi, [x1 x2], 1);

    % Create grid graph
    disp("  Processing grid graph")
    sampleResolution = (size(X1,1));
    g = nDGridGraph(2, sampleResolution);
    scores = predictManNoTransform(bias, weights, x_vals);
    y_vals = sign(scores);
    scoreGrid = reshape(y_vals(:),size(X1,1),size(X2,2));

    % Remove edges between nodes with different labels
    g = cutGraphFromPrediction(g, y_vals);

    
    % Create a graph object from the extracted edges
    g_del = graph(edges(:, 1), edges(:, 2));
    
    scores = predictManNoTransform(bias, weights, x_delaunay_feature);
    y_vals = sign(scores);

    % Remove edges between nodes with different labels
    g_del = cutGraphFromPrediction(g_del, y_vals);
    
    disp("  Calculating performance measures")
    % Count connected components in graph
    numComponents = countConnectedComponents(g);
    numComponents_del = countConnectedComponents(g_del);

    % Calculate Accuracy on test dataset
    acc = accuracy(YTest, sign(predictManNoTransform(bias, weights, XTestFeature)));

    % Create figure
    figure
    hold on
    contourf(X1,X2,scoreGrid,1)

    map = [255 115 115
        255 115 115
        131 115 255
        131 115 255] / 255;
    colormap(map)
    
    scatter(Xcp(Ycp == 1,1),  Xcp(Ycp == 1,2), 'b', 'o', 'filled');
    scatter(Xcp(Ycp == -1,1), Xcp(Ycp == -1,2), 'r', 'x');
    scatter(cps(:,1), cps(:,2), 'black', 'pentagram', 'filled')
    
    svInd = cls.IsSupportVector;
    plot(Xcp(svInd,1),X(svInd,2),'o','MarkerSize',10)
    colorbar;

    xlabel("x")
    ylabel("y")

    title("AACP, Critical Point: x=" + round(cp(1),3) + ", y=" + round(cp(2),3) + ", β_0=" + numComponents + ", accuracy=" + acc);
    legend('Prediction', 'Class 1', 'Class -1', 'Critical Points', 'Support Vectors');
    % Print LaTeX style table:
    % disp("$("+cp(1)+","+cp(2)+")$ & $"+numComponents+"$ & $"+numComponents_del+"$ & $"+acc+"$\%\\");
    hold off;
end


function [X,Y] = createDataset(n)
    centerX = .5;
    centerY = .5;
    radius = 0.5;
    differenceRadius = 0.25;
    
    X = zeros(n,2);
    Y = ones(n,1);
    
    for i = 1:n/2
        % Generate a random angle within the circle
        theta = 2 * pi * rand();
        
        % Generate a random radius within the circle
        r = radius * sqrt(rand());
        
        % Calculate the coordinates of the random point
        X(i,1) = centerX + r * cos(theta);
        X(i,2) = centerY + r * sin(theta);
    end

    for i = n/2 + 1:n
        Y(i) = -1;

        % Generate a random angle within the circle
        theta = 2 * pi * rand();
        
        % Generate a random radius within the circle
        r = radius * sqrt(rand()) + differenceRadius;
        
        % Calculate the coordinates of the random point
        X(i,1) = centerX + r * cos(theta);
        X(i,2) = centerY + r * sin(theta);
    end
end
