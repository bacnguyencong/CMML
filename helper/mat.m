function V = mat(V)
% ===================================================
% Get matrix form (d x d) for V
% 
% Created by: Bac Nguyen (Bac.NguyenCong@ugent.be)
% Date      : November 6, 2016
% ===================================================
    r=round(sqrt(length(V)));
    V=reshape(V,r,r);
end