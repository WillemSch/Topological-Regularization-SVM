function score = accuracy(Y, YPred)
    correct = 0;
    for i = 1:size(Y,1)
        if Y(i) == YPred(i)
            correct = correct + 1;
        end
    end
    score = correct/size(Y,1);
end

