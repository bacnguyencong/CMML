clear all; clc; rng('default');

addpath(genpath('helper/'));
addpath(genpath('solver/'));
data =importdata('data/iris.txt');

XTr = data(:,1:end-1)';
YTr = data(:,end);
num_cls = 3;
knn = 3;
k1  = knn;
k2  = knn;

params.num_cls   = num_cls; % number of clustering
params.alpha     = 1e-1;    % hyper-parameter alpha
params.beta      = 1e-2;    % hyper-parameter beta
params.max_iters = 20000;   % number of iterations
params.knn       = knn;
params.tol       = 1e-6;
params.quiet     = 1;
params.solver    = 3; % batch-GD (1), SGD (2), C++ SGD (3)

% initialize
fprintf('Running k-means clustering...\n');
[clusters, centers] = kmeans(XTr', num_cls); 
centers = centers';

% finding constraints
fprintf('Finding constraints ...\n');
Const = getAllConstraints(XTr, YTr, k1, k2);
T = cell(3, 1);
for c = 1:num_cls,
    T{c} = Const(:, clusters(Const(1,:)) == c);
end

% Euclidean distance metric
acc0 = 100*mean(YTr==loo_single_metric(eye(size(XTr, 1)), XTr, YTr, knn));

% run the algorithm
[A, cost] = CMML(XTr, T, params);
acc1 = 100*mean(YTr==loo_mult_metric(A, XTr, YTr, XTr, knn, centers));

fprintf('Euclidean = %.2f, CMML = %.2f\n', acc0, acc1);


