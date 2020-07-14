function [OvRmod_final, params] = cv2_OvRnFSGLASSO(Xs_trSet, Ys_trSet, ...
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
    
    n_task = length(Xs_trSet);
    n_class = length(unique(Ys_trSet{1}));
    [m, n_features] = size(Xs_trSet{1});
    params.Rho1 = zeros(1, 1);
    params.Rho2 = zeros(1, 1);
    [mRho, n] = size(Rhos);
    AUC_cv = zeros(mRho, n_task, cv_fold);
    aucs = zeros(mRho, n_class, n_task, cv_fold);
    
    for rho = 1:mRho
        
        for fold = 1:cv_fold
            % for tracking progress
            fileID = fopen('progress.txt','a');
            fprintf(fileID,'%1s\n','fold');
            fclose(fileID);
            
            cv_Xtr = cell(1, n_task);
            cv_Ytr = cell(1, n_task);
            cv_Xval = cell(1, n_task);
            cv_Yval = cell(1, n_task);
            
            for t = 1:n_task
                rng(10); %for repeatability
                cv = cvpartition(Ys_trSet{t}, 'k', cv_fold, 'Stratify', true);
                tr_idx = cv.training(fold);
                val_idx = cv.test(fold);
                
                cv_Xtr{t} = Xs_trSet{t}(tr_idx, :);
                cv_Ytr{t} = Ys_trSet{t}(tr_idx, :);
                cv_Xval{t} = Xs_trSet{t}(val_idx, :);
                cv_Yval{t} = Ys_trSet{t}(val_idx, :);
            end
            
            OvRMod = OvRTrain_nFSGLASSO(cv_Xtr, cv_Ytr, Rhos(rho, :), opts);
            Ys_pred = OvRPredict(cv_Xval, OvRMod);
            for t = 1:n_task
                [aucs_, AUC_cv(rho, t, fold)] = multiclassAUC(Ys_pred{t}, cv_Yval{t});
                aucs(rho, :, t, fold) = cell2mat(aucs_);
            end
        end
    end

    % Collecting weights and c from the best set of params
    AUC_avg_ = mean(AUC_cv, 3);
    AUC_avg = mean(AUC_avg_, 2);
    [M, I] = max(AUC_avg);
    
    OvRMod = OvRTrain_nFSGLASSO(cv_Xtr, cv_Ytr, Rhos(I, :), opts);
    params.Rho1 = Rhos(I, 1);
    params.Rho2 = Rhos(I, 2);

    W_final = cell(1, n_class);
    c_final = cell(1, n_class);
    for n = 1:n_class
        W_final{1, n}(:, :) = OvRMod(n).W;
        c_final{1, n}(:) = OvRMod(n).c;
    end      
    labelSet = OvRMod(1).labelSet;
    
    OvRmod_final = struct('W', W_final, 'c', c_final, 'labelSet', labelSet);
end
