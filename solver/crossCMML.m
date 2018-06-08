function [M, clusters, X, Y, centers] = crossCMML(XTr, YTr, params)
%%  Cross validate to find the best hyper-parameters for CMML
%   INPUT:
%       XTr: (d x n) training examples
%       YTr: (n x 1) class labels
    
    if size(XTr,2) > 5000,
        CVO = cvpartition(YTr,'HoldOut', 5000/size(XTr,2));
        XTe = XTr(:, CVO.test); YTe = YTr(CVO.test);
    else
        XTe = XTr; YTe = YTr;
    end
    
    pars    = params.par;      % input parameters
    k       = params.par.knn;  % number of neighbors  
    k1      = pars.k1;         % number of positive examples
    k2      = pars.k2;         % number of negative examples
    num_cls = pars.num_cls;    % number of clusters    
    T       = cell(num_cls, 1);% constraints;
    X       = cell(num_cls, 1);% training examples per cluster
    Y       = cell(num_cls, 1);% class labels per cluster
    
    % initialize
    fprintf('Running k-means clustering...\n');
    
    % using fast k-means
    [clusters, centers] = kmeans(XTr', num_cls);
    centers = centers';
    
    % finding constraints
    fprintf('Finding constraints ...\n');
    Const = getAllConstraints(XTr, YTr, k1, k2);
    for c = 1:num_cls,
        X{c} = XTr(:,clusters == c);
        Y{c} = YTr(clusters == c);
        T{c} = Const(:, clusters(Const(1,:)) == c);
    end
    
    fprintf('Alpha\t Beta\t Accuracy\t Cost\n');
    fprintf('-------------------------------------\n');
    bestAcc = -Inf;
      
    for alpha = 10.^(-3:2:1),
        for beta = 10.^(-3:-1),
            
            pars.alpha = alpha;
            pars.beta = beta;
            
            % run the algorithm
            [A, cost] = CMML(XTr, T, pars);
            accTemp = 100*mean(YTe==loo_mult_metric(A, XTr, YTr, XTe, k,centers));
            
            fprintf('%.3f\t%.3f\t%.2f\t%.5f\n', alpha, beta, accTemp, cost);
            
            % save the best result
            if (accTemp > bestAcc),
                M = A;
                bestAcc = accTemp;
            end
        end
    end
    fprintf('-------------------------------------\n');
    
end



