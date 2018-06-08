function cost = cost_func(M, X, T, beta, M0)
% compute the original cost function
    [~, Lx] = proj_psd(M);
    Lx = Lx'*X;
    cost = max (0, ...
                    1 + sum((Lx(:,T(1,:)) - Lx(:,T(2,:))).^2, 1)...
                      - sum((Lx(:,T(1,:)) - Lx(:,T(3,:))).^2, 1)...
                 );
    if size(T,2)
        cost = 0.5*beta*sum(vec((M - M0).^2)) + mean(cost);
    else
        cost = 0.5*beta*sum(vec((M - M0).^2));
    end
end
