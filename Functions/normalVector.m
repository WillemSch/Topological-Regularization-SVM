function weights = normalVector(cls, featureVector, symbols)
    % Transform all support vectors to the feature space
    svsFeatures = zeros(size(cls.SupportVectors,1), size(featureVector,2));
    for i = 1:size(cls.SupportVectors,1)
        svsFeatures(i,:) = subs(featureVector, symbols, [cls.SupportVectors(i,:)]);
    end
    
    % Calculate the normal vector of the linear separator of cls
    labeledAlpha = cls.Alpha .* cls.SupportVectorLabels;
    weights = mtimes(svsFeatures.', labeledAlpha);
end

