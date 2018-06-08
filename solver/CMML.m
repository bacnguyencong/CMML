function [M, curr_cost] = CMML(XTr, T, params)
%% Clustered multi-metric learning
%   INPUT:
%       XTr: (d x n) training set
%       T{c}: (3 x m) set of triplets (i, j, l), one per column
%       params: .num_cls : the number of clusters
%               .alpha   : the parameter alpha
%               .beta    : the parameter beta
%               .tol     : the tolerance of the algorithm
%               .solver  : (1) Batch GD (2) Stochastic GD in matlab (3) Stochastic GD in C
%               .quiet   : set to 0 to see the output
%   OUTPUT:
%       M:  a cell containing different matrices M{1}, .., M{end}: M0
%       curr_cost: the cost value
%
%   AUTHOR: Bac Nguyen (Bac.NguyenCong@ugent.be)
%   DATE: November 30, 2017
%

    num_cls = params.num_cls; % number of clusters
    alpha   = params.alpha;   % alpha
    beta    = params.beta;    % beta
    tol     = params.tol;     % tolerance
    d       = size(XTr, 1);
    
    % initial matrices, the last matrix denotes M_0
    [M_tmp{1:num_cls+1, 1}] = deal(eye(d));

    % begin algorithm
    prev_cost = Inf;          % the previuos cost value
    
    if ~params.quiet,
        fprintf('#iter=%.0f, C=%.8f\n', 0, get_cost(XTr, M_tmp, T, alpha, beta, num_cls));
    end
    
    for iter = 1:10,
        S = zeros(d);
        % solving for each subproblem
        for c = 1:num_cls,
            if size(T{c},2),
                % selecting el solver 
                switch params.solver, 
                    case 1
                        M_tmp{c} = subGD(XTr,    T{c}, M_tmp{end}, params); % batch learning
                    case 2 
                        M_tmp{c} = stoGD(XTr,    T{c}, M_tmp{end}, params); % matlab online                
                    case 3 
                        M_tmp{c} = mexStoGD(XTr, T{c}, M_tmp{end}, params); % C mex online
                end
            else
                M_tmp{c} = M_tmp{end};
            end
            S = S + M_tmp{c};
        end
        
        % solving for the common matrix
        M_tmp{end} = proj_psd(S/num_cls - (alpha/beta)*eye(d));
        curr_cost_tmp = get_cost(XTr, M_tmp, T, alpha, beta, num_cls);
        if ~params.quiet,
            fprintf('#iter=%.0f, C=%.8f\n', iter, curr_cost_tmp);
        end
        
        if prev_cost - curr_cost_tmp < tol,
            break;
        end
        
        % saving the previous cost
        prev_cost = curr_cost_tmp;
        
        % saving the best result
        M = M_tmp;
        curr_cost = curr_cost_tmp;
       
    end    
end