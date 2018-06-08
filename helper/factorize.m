function L = factorize(M)
% ===================================================
% Compute mapped data
% INPUT   M: (d x d) input matrix
% OUTPUT  L: L*L' = M
% 
% Created by: Bac Nguyen (Bac.NguyenCong@ugent.be)
% Date      : November 6, 2016
% ===================================================
    [L, S] = svd(M);
    L = bsxfun(@times, sqrt(diag(S)), L')';
end
