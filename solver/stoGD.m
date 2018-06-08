function [M, curr_cost, f_cost] = stoGD(XTr, T, M0, params)

    if (size(T,2) == 0)
        M = M0;
        curr_cost = 0;
        f_cost = 0;
        return;
    end
    
    beta = params.beta;
    quiet = params.quiet;
    
    f_cost = zeros(10, 1);
    %random permuting the input constraints
    index = randperm(size(T,2));
    M = M0;
    iter = 1;
    
    for epoch=1:10,
        for t = index,
            i = T(1,t); j = T(2,t); l = T(3,t);
            u = XTr(:,i) - XTr(:,j);
            v = XTr(:,i) - XTr(:,l);
            up = 1 - v'*M*v + u'*M*u;
            eta = 1/(beta*iter);
            if up > 0,
                Z = v*v'-u*u';
                M = proj_psd((1.0 - 1.0/iter)*M + 1.0/iter*M0 + eta*Z);
            else
                M = (1.0 - 1.0/iter)*M + 1.0/iter*M0;               
            end
            iter = iter + 1;
        end
        
        curr_cost = cost_func(M, XTr, T, beta, M0);
        f_cost(epoch) = curr_cost;
        if (~quiet),
            fprintf('#epoch=%.0f, C=%.6f\n', epoch, curr_cost); 
        end        
    end
    curr_cost = cost_func(M, XTr, T, beta, M0);
end
