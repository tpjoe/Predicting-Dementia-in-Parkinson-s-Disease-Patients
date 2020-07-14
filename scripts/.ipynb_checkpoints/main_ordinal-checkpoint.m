%% setwd
%% cd ~/tpjoe@stanford.edu/project_Udall2
addpath('../libraries/progress_bar'); 
% poolObj = parpool(36);
%% Load X, Y
xydir = "../data";
% Xlead = 'X_yearsince1stvisit_stdScaled.';
% Ylead = 'y_yearsince1stvisit_stdScaled.';
% max_task = "max";
[Xs, Ys] = getXY(xydir, Xlead, Ylead, max_task);

%% model params

% For TGL
% Rho1 = [0.00001];%, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025];%, 0.0005, 0.01];
% Rho2 = [0.0063];%, 0.0125, 0.025, 0.05];
% Rho3 = [0.00025];%, 0.0005, 0.00075, 0.001];%, 0.0001, 0.00025, 0.0005, 0.01];

% For CFGL
% Rho1 = [0.00005]%, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01];
% Rho2 = [0.005]%, 0.01, 0.025, 0.05, 0.1];
% Rho3 = [0.0001]%, 0.00025,0.0005, 0.001, 0.0025, 0.005, 0.01]
%
% For nFSGL
% Rho1 = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5];
% Rho2 = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5];
% 
% iter = 36;
% cvfold = 3;
% cv_method = "cv";
% model_type = "TGL"; 
% test_frac = 0.25;
% outer_cv_method = "stratified holdout";

if model_type ~= "nFSGL"
    Rhos = combvec(Rho1, Rho2, Rho3)';
    opts.tol = 10^-6; % tolerance
else
    Rhos = combvec(Rho1, Rho2)';
    opts.tol = 10; % tolerance
end
opts.init = 2; % compute start point from data
opts.tFlag = 1; % terminate after relative objective
opts.maxIter = 10000; % maximum iteration number of optimization.

%% Initiating recallable variables
n_task = length(Xs);
n_class = length(unique(Ys{1}));
macro_AUC = zeros(n_task, iter);
AUCs = zeros(n_task, n_class, iter);
if cv_method == "cv2"
    param_Rho1 = zeros(1, iter); % Best Rho1 from each cv of each iter.
    param_Rho2 = zeros(1, iter); % Best Rho2 from each cv of each iter.
    param_Rho3 = zeros(1, iter); % Best Rho3 from each cv of each iter.
else
    param_Rho1 = zeros(cvfold, iter); % Best Rho1 from each cv of each iter.
    param_Rho2 = zeros(cvfold, iter); % Best Rho2 from each cv of each iter.
    param_Rho3 = zeros(cvfold, iter); % Best Rho3 from each cv of each iter.
end
W = cell(n_class, iter); % W returns the stored weights from all iterations with W{n_class, iter} = n_features x n_task matrix.
c = cell(n_class, iter); % c returns the stored weights from all iterations with c{n_class, iter} = 1 x n_task matrix.
Ys_pred = cell(n_task, iter); % Ys_pred is a cell with {n_task, iter} dimension each cell with array of {n_samples, n_class} dimension.
Ys_test = cell(n_task, iter); % Same as Ys_pred, this is for collecting Ys_test in each iteration, so we can reference later.
W_mean = cell(n_class, 1); % Mean W with as a cell of dimension {n_lcass, 1} where each cell has a dimension of {n_feature, n_task}.
c_mean = cell(n_class, 1); % Mean c with as a cell of dimension {n_lcass, 1} where each cell has a dimension of {1, n_task}.
test_size = zeros(1, n_task); % Number of test samples in each task.
for t = 1:n_task
    test_size(1, t) = cvpartition(Ys{t}, 'holdout', test_frac, 'Stratify', ...
    true).TestSize;
end

%% Run model

fileID = fopen('progress.txt','w');
fclose(fileID);

[m, n_task] = size(Xs);
n_class = length(unique(Ys{1}));
[m, n_feature] = size(Xs{1});
% WaitMessage = parfor_wait(iter, 'Waitbar', true);


tic
parfor (i = 1:iter, 36)
    Xs_trSet = cell(1, n_task);
    Ys_trSet = cell(1, n_task);
    Xs_teSet = cell(1, n_task);
    Ys_teSet = cell(1, n_task);

    for t = 1:n_task
        min_test = 0;
        unique_y_test = 0;
        unique_y_train = 0;
        ii = i;
        while min_test < 3 || unique_y_test~= 3 || unique_y_train~=3
            if outer_cv_method == "stratified holdout"
                rng(ii*100);
                cv_ = cvpartition(Ys{t}, 'holdout', test_frac, 'Stratify', true);
                trSet_idx = find(cv_.training());
                teSet_idx = find(cv_.test());

            elseif outer_cv_method == "capped weighted sampling"           
                %weighted sampling is capped at 75% where the rest goes
                %to the highest number of samples
                weights = zeros(1, length(Ys{t}));
                all_counts =  length(Ys{t});
                classes = unique(Ys{t});
                est_test_each = test_frac*all_counts/n_class;
                for n = 1:n_class
                    class_count = sum(Ys{t} == classes(n));
                    weights(Ys{t} == classes(n)) = (all_counts/class_count);
                    max_test = 0.70*class_count;  %adjusting ratio to max at 0.7
                    if est_test_each > max_test
                        weights(Ys{t} == classes(n)) = weights(Ys{t} == ...
                            classes(n)).*(max_test/est_test_each);
                    end
                end
                rng(ii);
                [y, tr_idx] = datasample(1:length(Ys{t}), round(length(Ys{t})*test_frac), ...
                    'Replace', false, 'Weights', weights);
                trSet_idx = sort(tr_idx);
                teSet_idx = setdiff(1:length(Ys{t}), tr_idx);
            end        
            Xs_trSet{t} = Xs{t}(trSet_idx, :);
            Ys_trSet{t} = Ys{t}(trSet_idx, :);
            Xs_teSet{t} = Xs{t}(teSet_idx, :);
            Ys_teSet{t} = Ys{t}(teSet_idx, :);
            
            %make sure that min number of test samples >= 3
            test_sizes = tabulate(Ys_teSet{t});
            min_test = min(test_sizes(:, 2));
            unique_y_test = size(test_sizes, 1); %number of row = unique y
            unique_y_train = size(tabulate(Ys_trSet{t}), 1);
            ii = ii + 100;
        end
    end
    Ys_test(:, i) = Ys_teSet';
    
%     for visit = 1:length(Ys_trSet)
%     display([tabulate(Ys_teSet{visit}), tabulate(Ys_trSet{visit})])
%     end
    
    if cv_method == "cv2"
        if model_type == "TGL"
            [OrdinalMod, p] = cv2_OvRTemporalLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, cv_method, opts);
        elseif model_type == "CFGL"
            [OrdinalMod, p] = cv2_OvRCFGLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, cv_method, opts);
        elseif model_type == "nFSGL"
            [OrdinalMod, p] = cv2_OvRnFSGLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, cv_method, opts);
        end
    else
        if model_type == "TGL"
            [OrdinalMod, p] = cv_ordinalTemporalLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, cv_method, opts);
        elseif model_type == "CFGL"
            [OrdinalMod, p] = cv_OvRCFGLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, cv_method, opts);
        elseif model_type == "nFSGL"
            [OrdinalMod, p] = cv_OvRnFSGLASSO(Xs_trSet, Ys_trSet, cvfold, Rhos, cv_method, opts);
        end
    end
    
    param_Rho1(:, i) = p.Rho1;
    param_Rho2(:, i) = p.Rho2;
    if model_type ~= "nFSGL"
        param_Rho3(:, i) = p.Rho3;
    end
        
    Ys_pred_ = ordinalPredict(Xs_teSet, OrdinalMod);
    Ys_pred(:, i) = Ys_pred_';
    
    % collect weights
    W_ = cell(n_class, 1);
    c_ = cell(n_class, 1);
    for n = 1:n_class
        % store W
        W_{n, 1} = OrdinalMod(n).W;
        c_{n, 1} = OrdinalMod(n).c;
    end    
    W(:, i) = W_;
    c(:, i) = c_;
        
    % calculate AUC
    macro_AUC_ = zeros(n_task, 1);
    AUCs_ = zeros(n_task, n_class, 1);
    for t = 1:n_task      
        % calculate AUCs
        [aucs, macro_AUC_(t, 1), microauc] = multiclassAUC(Ys_pred_{t}, Ys_teSet{t});
        AUCs_(t, :, 1) = cell2mat(aucs);
    end
    
    AUCs(:, :, i) = AUCs_;
    macro_AUC(:, i) = macro_AUC_;
    
%         WaitMessage.Send; 
end
toc

% WaitMessage.Destroy

%% Convert Ys to cell matrix for Python

Ys_pred_mat = cell2mat(Ys_pred);
Ys_test_mat = cell2mat(Ys_test);

%% Calculate mean weights
W_mat = cell2mat(W);
[n_classxn_feature, n_iterxn_ntask] = size(W_mat);
for n = 1:n_class   
    W_mean{n} = zeros(n_feature, n_task);
    c_mean{n} = zeros(1, n_task);
    W_class = cell2mat(W(n, :));
    c_class = cell2mat(c(n, :));
    for t = 1:n_task
        selected_column = t:n_task:n_iterxn_ntask;
        W_mean{n}(:, t) = mean(W_class(:, selected_column), 2);
        c_mean{n}(:, t) = mean(c_class(:, selected_column), 2);
    end 
end
%%
% delete(poolObj)