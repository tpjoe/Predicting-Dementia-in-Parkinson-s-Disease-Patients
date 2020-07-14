function [W_final, c_final, params] = cvTemporalLASSO(Xs_trSet, Ys_trSet, ...
    cv_fold, Rhos, opts)

    % add relavant paths
    addpath('../libraries/MALSAR/functions/Lasso/'); 
    addpath('../libraries/MALSAR/utils/'); 
    addpath(genpath('../libraries/MALSAR/c_files/')); 
    addpath('../libraries/MALSAR/functions/progression_model/TGL');
    
    if nargin < 5
        cv_fold = 5;
    end
    
    n_task = length(Xs_trSet);
    [m, n_features] = size(Xs_trSet{1});
    params.Rho1 = zeros(1, cv_fold);
    params.Rho2 = zeros(1, cv_fold);
    params.Rho3 = zeros(1, cv_fold);

    for fold = 1:cv_fold
            cv_Xtr = cell(1, n_task);
            cv_Ytr = cell(1, n_task);
            cv_Xval = cell(1, n_task);
            cv_Yval = cell(1, n_task);
            [mRho, n] = size(Rhos);
            AUC_rho = zeros(1, mRho);
            W_cv = zeros(n_features, n_task, cv_fold);
            c_cv = zeros(n_task, cv_fold);

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
            
            for rho = 1:mRho
                [W, c, funcVal] = Logistic_TGL(cv_Xtr, cv_Ytr,...
                    Rhos(rho, 1), Rhos(rho, 2), Rhos(rho, 3), opts);
                AUC = zeros(1, n_task);

                for visit = 1:n_task
                    prob = cv_Xval{visit} * W(:, visit) + c(visit);
                    y_pred = (1./(1 + exp(1).^(-1*prob)));
                    [xROC, yROC, T, auc] = perfcurve(cv_Yval{visit}, y_pred, 1);
                    AUC(1, visit) = auc;
                end         
                AUC_rho(1, rho) = mean(AUC);
            end        


            [maxAUC, maxAUCIdx] = max(AUC_rho);
            bestRho1 = Rhos(maxAUCIdx, 1);
            bestRho2 = Rhos(maxAUCIdx, 2);
            bestRho3 = Rhos(maxAUCIdx, 3);
            params.Rho1(1, fold) = bestRho1;
            params.Rho2(1, fold) = bestRho2;
            params.Rho3(1, fold) = bestRho3;
            [W, c, funcVal] = Logistic_TGL(cv_Xtr, cv_Ytr,...
                bestRho1, bestRho2, bestRho3, opts);
            W_cv(:, :, fold) = W;
            c_cv(:, fold) = c;
            
    end

    W_final = mean(W_cv, 3);
    c_final = mean(c_cv, 2);

end