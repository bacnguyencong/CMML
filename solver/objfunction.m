function [ val, grad, slack ] = objfunction(x, pars, Lx, beta, grad, old_slack)
% =============================================================================
% Compute the objective function and its subgradient at point x
% INPUT 
%       x: the matrix parameter
%       pars
%            X: (d x n) input examples
%            alpha: the hyper-parameter 
%            T: triplet constraints    
%       Lx:(m x n) input examples in the transformed space
%       beta: the weight on loss function
%       grad: (d^2 x 1) the previous gradient 
%       old_slack: (1 x m) the previous slack variables
% OUTPUT
%       val:  the value of objective function at x
%       grad:(d x 1)   the gradient value
%       slack: (1 x m) the slack varibles
% 
% Created by: Bac Nguyen (Bac.NguyenCong@ugent.be)
% Date      : November 6, 2016
% =============================================================================
    slack = max (0, ...
                    1 + sum((Lx(:,pars.T(1,:)) - Lx(:,pars.T(2,:))).^2, 1)...
                      - sum((Lx(:,pars.T(1,:)) - Lx(:,pars.T(3,:))).^2, 1)...
                 );
    % the value of objective function at x
    if (~pars.kernel)
        val = pars.alpha*trace(mat(x))    + beta*sum(slack);
    else
        val = pars.alpha*(x'*vec(pars.X)) + beta*sum(slack);
    end
    % adding new items
    ind  = find(slack > 0 & old_slack == 0);    
    grad = grad + beta*vec(SOPD(pars.X, pars.T(1,ind), pars.T(2,ind)))...
                - beta*vec(SOPD(pars.X, pars.T(1,ind), pars.T(3,ind)));
    % removing old items
    ind  = find(slack == 0 & old_slack > 0);    
    grad = grad - beta*vec(SOPD(pars.X, pars.T(1,ind), pars.T(2,ind)))...
                + beta*vec(SOPD(pars.X, pars.T(1,ind), pars.T(3,ind)));    
end
