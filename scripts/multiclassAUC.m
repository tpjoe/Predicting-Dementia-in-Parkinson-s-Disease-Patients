function [AUCs, AUC_macro, AUC_micro] = multiclassAUC(y_pred, y_test)    
    classes = unique(y_test);
    n_class = length(classes);
%     fpr = cell(1, n_class);
%     tpr = cell(1, n_class);
    AUCs = cell(1, n_class);
    
    % Calculating individual auc
    for n = 1:n_class
        [f, t, T, AUCs{n}] = perfcurve(y_test, y_pred(:, n), classes(n));
    end
     
    % Calculating micro auc
    encoded_y_test = bsxfun(@eq, y_test(:), min(classes):max(classes));
    [tpr, fpr, thresholds] = roc(encoded_y_test', y_pred');
    [f, t, T, AUC_micro] = perfcurve( ...
        reshape(encoded_y_test.', 1, [])', reshape(y_pred.', 1, [])', 1);
    
    % Combining all fpr for macro_fpr
    all_fpr = unique([fpr{:}]');
    mean_tpr = zeros(size(all_fpr));
    
    for n = 1:n_class
        [fpr_, idx] = unique(fpr{n}, 'last');
        mean_tpr = mean_tpr + interp1(fpr_, tpr{n}(idx), all_fpr);
    end
    mean_tpr = mean_tpr / n_class;
    
    fpr_macro = all_fpr;
    tpr_macro = mean_tpr;
    AUC_macro = trapz(fpr_macro, tpr_macro);
end
