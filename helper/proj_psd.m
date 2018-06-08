function [M, L] = proj_psd(M)
    [L, D] = eig(M); L = real(L); D = real(D);
    ind    = find(diag(D) > 0);
    M      = L(:,ind) * D(ind, ind) * L(:,ind)';
    L      = L(:,ind) * sqrt(D(ind, ind));
end