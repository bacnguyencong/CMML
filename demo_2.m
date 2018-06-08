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
params.max_iters = 20000;   % number of iterations
params.knn       = knn;
params.k1        = k1;
params.k2        = k2;
params.tol       = 1e-6;
params.quiet     = 1;
params.solver    = 3; % batch-GD (1), SGD (2), C++ SGD (3)

pars.par = params;

% Euclidean distance metric
acc0 = 100*mean(YTr==loo_single_metric(eye(size(XTr, 1)), XTr, YTr, knn));

% Clustered multi-metric learning
[M, clusters, ~, ~, centers] = crossCMML(XTr, YTr, pars);
acc1 = 100*mean(YTr==loo_mult_metric(M, XTr, YTr, XTr, knn, centers));


fprintf('LOO-accuracy of kNN: \nEuclidean = %.2f, \nCMML = %.2f\n', acc0, acc1);


