function weights = normalVector(cls, featureVector, symbols, kernelScale)
    % Transform all support vectors to the feature space
    if size(cls.SupportVectors,2) == size(symbols,1)
        svsFeatures = zeros(size(cls.SupportVectors,1), size(featureVector,2));
        for i = 1:size(cls.SupportVectors,1)
            svsFeatures(i,:) = subs(featureVector, symbols, [cls.SupportVectors(i,:)]/kernelScale);
        end
    else
        svsFeatures = cls.SupportVectors;
    end

    % Calculate the normal vector of the linear separator of cls
    labeledAlpha = cls.Alpha .* cls.SupportVectorLabels;
    weights = mtimes(svsFeatures.', labeledAlpha);
end

