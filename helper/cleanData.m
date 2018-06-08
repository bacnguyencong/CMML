function index = cleanData(Y, k)
% ===================================================
% Removing examples with small k
% INPUT 
%       Y: (n x 1) input labels
%       k: number of target neighbors
% OUTPUT  
%       index: (m x 1) indices of active examples
% 
% Created by: Bac Nguyen (Bac.NguyenCong@ugent.be)
% Date      : November 6, 2016
% ===================================================

    labels = unique(Y);
    index  = true(length(Y), 1);
    for i=1:length(labels)
        ind = (Y == labels(i));
        if (sum(ind) <= k)            
            index(ind) = 0;
            fprintf('Removing class (%d)\n', labels(i));
        end
    end
    index = vec(find(index));
end