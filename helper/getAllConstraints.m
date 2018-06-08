function T = getAllConstraints(X, Y, k, v)
%%=========================================================================
% Find target neighbours for each instance in the training set impostors
% 
% INPUT:
%       X : (d x n) training examples by columns   
%       Y : (n x 1) labels for each example by column
%       k : number of target neighbors
%       v : number of nearest neighbors of different class
% OUPUT:
%       T : (3 x m) index of triplet constraints
%
% Created by: Bac Nguyen (Bac.NguyenCong@ugent.be)
% Date      : November 6, 2016
% =========================================================================

    n = size(X,2);
    l = unique(Y);
    
    D = sum(X.^2,1);
    D = bsxfun(@plus,D,bsxfun(@plus,D',-2*(X'*X)));
    D(1:n+1:end) = inf;
    
    T = zeros(3,n*(v+k)*(v+k)); % targets for training    
    m = 0;
    
    for i=1:length(l)
        %fprintf('Searching constraints in class (%d)\n', l(i));
        
        %%% find targets
        inds = find(Y == l(i));
        k1   = min(length(inds)-1,k);
        if k1 < 1, continue; end;
        
        [~,tars] = sort(D(inds,inds),1);
        tars     = inds(tars(1:k1,:));
        
        %%% find impostors
        indd = find(Y ~= l(i));
        k2 = min(length(indd),v);
        if k2 < 1, continue; end;
        
        [~,imps] = sort(D(indd,inds),1);
        imps     = indd(imps(1:k2,:));
        
        %%% adding triplet constraints
        [C, len] = joinTriplets(inds,Y,tars,imps,k1,k2);        
        T(:,m+1:m+len) = C;
        m = m + len;
    end   
    T = T(:,1:m);
    %fprintf('------------------------------------------------------\n');
end

function [T, len] = joinTriplets(inds, Y, tars, imps, k1, k2)
    inds = vec(inds)';
    n = length(inds);
    T = zeros(3,n*k2*k2);
    T(1,:) = vec(repmat(inds,k2*k2,1));
    T(2,:) = vec(repmat(vec(imps),1,k2)');
    T(3,:) = vec(repmat(imps,k2,1));
    
    %first set of constraints
    G   = Y(T);
    ind = (G(3,:) > G(2,:) & G(2,:) > G(1,:)) ...
         |(G(1,:) > G(2,:) & G(2,:) > G(3,:));
    len = sum(ind);              
    T(:,1:len) = T(:,ind);
    
    % second set of constraints
    T(1,len+1:len + n*k1*k2) = vec(repmat(inds,k1*k2,1));
    T(2,len+1:len + n*k1*k2) = vec(repmat(vec(tars),1,k2)');
    T(3,len+1:len + n*k1*k2) = vec(repmat(imps,k1,1));       
    
    T   = T(:,1:len + n*k1*k2);
    len = len + n*k1*k2;
end