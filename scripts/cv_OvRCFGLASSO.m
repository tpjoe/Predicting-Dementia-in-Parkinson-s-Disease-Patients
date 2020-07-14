function [OvRmod_final, params] = cv_OvRCFGLASSO(Xs_trSet, Ys_trSet, ...
    cv_fold, Rhos, cv_method, opts)

    % The function returns OvRmod_final which is a {3 x n_class} struct
    % where in each n_class there are:
    % W: [n_features x n_tasks] double
    % c: [1 x n_tasks] double
    % labelSet: a set of unique y
    
    % add relavant paths
    addpath('../libraries/MALSAR/functions/Lasso/'); 
    addpath('../libraries/MALSAR/utils/'); 
    addpath(genpath('../libraries/MALSAR/c_files/')); 
    addpath('../libraries/MALSAR/functions/progression_model/TGL');
    
    if nargin < 5
        cv_fold = 5;
    end
    
    n_task = length(Xs_trSet);
    n_class = length(unique(Ys_trSet{1}));
    [m, n_features] = size(Xs_trSet{1});
    params.Rho1 = zeros(1, cv_fold);
    params.Rho2 = zeros(1, cv_fold);
    params.Rho3 = zeros(1, cv_fold);

    for fold = 1:cv_fold
            % for tracking progress
            fileID = fopen('progress.txt','a');
            fprintf(fileID,'%1s\n','fold');
            fclose(fileID);
            
            cv_Xtr = cell(1, n_task);
            cv_Ytr = cell(1, n_task);
            cv_Xval = cell(1, n_task);
            cv_Yval = cell(1, n_task);
            [mRho, n] = size(Rhos);
            AUC_rho = zeros(1, mRho);
            W_cv = cell(1, n_class);
            c_cv = cell(1, n_class);
            for n = 1:n_class
                W_cv{1, n} = zeros(n_features, n_task, cv_fold);
                c_cv{1, n} = zeros(n_task, cv_fold);
            end
            mods = {};

            for t = 1:n_task
                if cv_method == "cv"
                rng(10); %for repeatability
                cv = cvpartition(Ys_trSet{t}, 'k', cv_fold, 'Stratify', true);
                tr_idx = cv.training(fold);
                val_idx = cv.test(fold);
                end
                
                if cv_method == "stratified sampling"
                    rng(fold); %for repeatability
                    cv = cvpartition(Ys_trSet{t}, 'holdout', 0.25, 'Stratify', true);
                    tr_idx = cv.training();
                    val_idx = cv.test();
                end
                
                if cv_method == "weighted sampling"
                    weights = zeros(1, length(Ys_trSet{t}));
                    all_counts =  length(Ys_trSet{t});
                    classes = unique(Ys_trSet{t});
                    for n = 1:n_class
                        weights(Ys_trSet{t} == classes(n)) = (all_counts/sum(Ys_trSet{t} == classes(n)));                                    
                    end    
                    [y, tr_idx] = datasample(1:length(Ys_trSet{t}), round(length(Ys_trSet{t})* 0.50), ...
                        'Replace', false, 'Weights', weights);
                    tr_idx = sort(tr_idx);
                    val_idx = setdiff(1:length(Ys_trSet{t}), tr_idx);
                end
                
                cv_Xtr{t} = Xs_trSet{t}(tr_idx, :);
                cv_Ytr{t} = Ys_trSet{t}(tr_idx, :);
                cv_Xval{t} = Xs_trSet{t}(val_idx, :);
                cv_Yval{t} = Ys_trSet{t}(val_idx, :);
            end
            
            for rho = 1:mRho
                OvRMod = OvRTrain_CFGLASSO(cv_Xtr, cv_Ytr, ...
                    Rhos(rho, :), opts);
                mods{1, rho} = OvRMod;
                Ys_pred = OvRPredict(cv_Xval, OvRMod);
                AUC = zeros(1, n_task);
                
                for visit = 1:n_task
                    [aucs, AUC(1, visit)] = multiclassAUC(Ys_pred{visit}, cv_Yval{visit});
                end         
                AUC_rho(1, rho) = mean(AUC);
            end


            [maxAUC, maxAUCIdx] = max(AUC_rho);
            params.Rho1(1, fold) = Rhos(maxAUCIdx, 1);
            params.Rho2(1, fold) = Rhos(maxAUCIdx, 2);
            params.Rho3(1, fold) = Rhos(maxAUCIdx, 3);
            for n = 1:n_class
                W_cv{1, n}(:, :, fold) = mods{1, maxAUCIdx}(n).W;
                c_cv{1, n}(:, fold) = mods{1, maxAUCIdx}(n).c;
            end        
    end
    
    W_final = {n_class};
    c_final = {n_class};
    
    for n = 1:n_class
        W_final{n} = mean(W_cv{1, n}, 3);
        c_final{n} = mean(c_cv{1, n}, 2)';
        labelSet = OvRMod(1).labelSet;
    end
  
    OvRmod_final = struct('W', W_final, 'c', c_final, 'labelSet', labelSet);
end

