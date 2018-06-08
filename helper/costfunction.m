function [ cost ] = costfunction( M, Lx, pars)
    slack = max (0, ...
                    1 + sum((Lx(:,pars.T(1,:)) - Lx(:,pars.T(2,:))).^2, 1)...
                      - sum((Lx(:,pars.T(1,:)) - Lx(:,pars.T(3,:))).^2, 1)...
                 );
    if (~pars.kernel)
        cost = pars.alpha * trace(M) + 1./size(pars.T,2) * sum (slack);
    else
        cost = pars.alpha * trace(pars.X*M) + 1./size(pars.T,2) * sum (slack);
    end
end

