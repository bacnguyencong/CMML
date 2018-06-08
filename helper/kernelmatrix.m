% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'lin','poly','rbf','sam'
%       X:      (d x n) input data
%       X2:     (d x m) test data
%       sigma:  width of the RBF kernel
%       b:      bias in the linear and polinomial kernel
%       d:      degree in the polynomial kernel
%
% Output:
%       K: kernel matrix

function K = kernelmatrix(ker,X,X2,sigma, b, d)

switch ker
    case 'lin'
        if exist('X2','var')
            K = X' * X2;
        else
            K = X' * X;
        end

    case 'poly'
        if exist('X2','var')
            K = (X' * X2 + b).^d;
        else
            K = (X' * X + b).^d;
        end

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);

        if isempty(X2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*(X'*X);
        else
            n2sq = sum(X2.^2,1);
            n2 = size(X2,2);
            D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
        end;
        K = exp(-D*sigma);

    case 'sam'
        if exist('X2','var');
            D = X'*X2;
        else
            D = X'*X;
        end
        K = exp(-acos(D).^2/(2*sigma^2));
        
    otherwise
        error(['Unsupported kernel ' ker])
end
