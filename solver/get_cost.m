function cost = get_cost(XTr, M, T, alpha, beta, num_cls)
    cost = 0;
    for c=1:num_cls,
        cost = cost + cost_func(M{c}, XTr, T{c}, beta, M{end});
    end
    cost = cost/num_cls + alpha*trace(M{end});
end