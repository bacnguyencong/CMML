function [data, test] = normalize(d, t)
%% Normalize data by columns vectors so that max is 1, and min is 0
mmin = min([d],[],1);
mmax = max([d],[],1);
d = double(d);
data = (d -repmat(mmin,size(d,1),1)).*...
         1./max(1e-9, repmat(mmax-mmin, size(d,1),1));
test = (t -repmat(mmin,size(t,1),1)).*...
         1./max(1e-9, repmat(mmax-mmin, size(t,1),1));
end