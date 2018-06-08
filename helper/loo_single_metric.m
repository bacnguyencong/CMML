function preds = loo_single_metric(M, X, Y, k)
%LOO_SINGLE_METRIC Summary of this function goes here
%   Detailed explanation goes here

    preds = kNearestNeighbors(X, X, k+1, M);
    preds = Y(preds(2:k+1,:));            
    if k > 1, preds = mode(preds, 1); end    
    preds = preds(:);
end

