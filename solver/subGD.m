function [M, cost, f_cost] = subGD(X, T, M0, params)
    
    if (size(T, 2) == 0)
        M = M0;
        cost = 0;
        f_cost = 0;
        return;        
    end
    
    x0 = vec(eye(size(X,1))/sqrt(params.beta * size(X,1)));
    params.T = T;
    params.M0 = M0;
    params.X = X;
    [M, cost, f_cost] = gradproj(x0,1000,@myFunction, params);
    M = mat(M);
    M = 0.5*(M + M');
end

