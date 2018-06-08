function [T, I] = getTargets(xTr, yTr, k, v)
%%=========================================================================
% Find target neighbours for each instance in the training set impostors
% 
% INPUT:
%       xTr : (d x n) training examples by columns   
%       yTr : (n x 1) labels for each example by column
%       k   : number of target neighbors
%       v   : number of nearest neighbors of different class
% OUPUT:
%       T   : (k x n) index of k target neighbors for each instance
%       I   : (v x n) index of v nearest neighbors of different class
%
% Created by: Bac Nguyen (Bac.NguyenCong@ugent.be)
% Date      : November 6, 2016
% =========================================================================

    nInst = size(xTr, 2);
    labels= unique(yTr);
    
    index = 1:nInst;
    
    T     = zeros(k, nInst); % targets for training    
    I     = zeros(v, nInst); % impostors for training    
    sort(labels);
    
    for i=1:length(labels)       
        fprintf('Finding target neighbors of class (%d)', labels(i));
        x              = xTr(:, yTr == labels(i)); 
        indi           = index(:, yTr == labels(i));
        iknn           = kNearestNeighbors(x, x, k+1);        
        T(:,indi)      = indi(iknn(2:k+1,:));       
        
        x1             = xTr(:, yTr ~= labels(i)); 
        indi1          = index(:, yTr ~= labels(i));
        iknn           = kNearestNeighbors(x1, x, v);
        I(:,indi)      = indi1(iknn);
        
        clear('x', 'indi', 'iknn', 'x1', 'indi1');
        fprintf('.\n');
    end   
end

