function [M, cost] = mexStoGD(X, T, M0, params)
    if size(T, 2),
        [M, cost] = mexSolverStoGD(full(X), int32(T), M0, params.beta, params.max_iters);
    else
        M = M0; cost = 0;
    end
end