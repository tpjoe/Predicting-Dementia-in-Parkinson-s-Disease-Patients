%% setwd
cd ~/tpjoe@stanford.edu/project_Udall2
addpath('../libraries/progress_bar'); 
poolObj = parpool(36);
%% Load X, Y
xydir = "../data/";
[Xs, Ys] = getXY(xydir);

%% model params
Rho1 = [0.00001, 0.000025, 0.00005, 0.000075];%, 0.0001, 0.00025, 0.0005, 0.01];
Rho2 = [0.0063, 0.0125, 0.025, 0.05];
Rho3 = [0.00025, 0.0005, 0.00075, 0.001];%, 0.0001, 0.00025, 0.0005, 0.01];
Rhos = combvec(Rho1, Rho2, Rho3)';
iter = 36;
cvfold = 5;

opts.init = 2; % compute start point from data
opts.tFlag = 1; % terminate after relative objective
opts.tol = 10^-6; % tolerance
opts.maxIter = 10000; % maximum iteration number of optimization.

%% Run model

[m, n_task] = size(Xs);
AUCs = zeros(1, iter);
best_prams = zeros(1, iter);
% W_all = zeros();
% c_all = zeros();
param_Rho1 = zeros(cvfold, iter);
param_Rho2 = zeros(cvfold, iter);
param_Rho3 = zeros(cvfold, iter);

WaitMessage = parfor_wait(iter, 'Waitbar', true);
% for i = 1:iter
tic
parfor (i = 1:iter, 36)    
    Xs_trSet = cell(1, n_task);
    Ys_trSet = cell(1, n_task);
    Xs_teSet = cell(1, n_task);
    Ys_teSet = cell(1, n_task);

    for t = 1:n_task
        rng(i)
        cv_ = cvpartition(Ys{t}, 'holdout', 0.50, 'Stratify', true);
        trSet_idx = cv_.training();
        teSet_idx = cv_.test();
        Xs_trSet{t} = Xs{t}(trSet_idx, :);
        Ys_trSet{t} = Ys{t}(trSet_idx, :);
        Xs_teSet{t} = Xs{t}(teSet_idx, :);
        Ys_teSet{t} = Ys{t}(teSet_idx, :);
    end

    [W_cv, c_cv, p] = cvTemporalLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, opts);
    param_Rho1(:, i) = p.Rho1;
    param_Rho2(:, i) = p.Rho2;
    param_Rho3(:, i) = p.Rho3;
    
    visit = 2;
    prob = Xs_teSet{visit} * W_cv(:, visit) + c_cv(visit);
    y_pred = (1./(1 + exp(1).^(-1*prob)));
    [xROC, yROC, T, AUC] = perfcurve(Ys_teSet{visit}, y_pred, 1);
    AUCs(1, i) = AUC;
    WaitMessage.Send;     
end
toc
WaitMessage.Destroy

%%
delete(poolObj)