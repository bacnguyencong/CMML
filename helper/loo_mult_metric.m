function  preds = loo_mult_metric(M, X, Y, XTe, k, centers)
%% Perform k nearest neigbor ckassification with LOO error

    % determine the center 
    clusters = kNearestNeighbors(centers, XTe, 1);
    
    num_cls  = length(X);
    preds    = zeros(size(XTe,2), 1);
    
    for c = 1:num_cls,
        index = clusters == c;
        if sum(index),
            temp = kNearestNeighbors(X, XTe(:,index), k+1, M{c});
            temp = Y(temp(2:k+1,:));
            if k > 1, temp = mode(temp, 1); end    
            preds(index) = temp(:);
        end
    end
    
end

