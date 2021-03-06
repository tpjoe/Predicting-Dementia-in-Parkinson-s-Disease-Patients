function [OvRmod] = OvRTrain_nFSGLASSO(Xs, Ys, rhos, opts)

    % Add path
    addpath('../libraries/MALSAR/functions/Lasso/'); 
    addpath('../libraries/MALSAR/utils/'); 
    addpath(genpath('../libraries/MALSAR/c_files/')); 
    addpath('../libraries/MALSAR/functions/progression_model/nFSGL/');
    
    % Check that all tasks contain at least one label of another class
    for i = 2:length(Ys)
        if length(unique(Ys{i})) ~= length(unique(Ys{1}))
            error("Inconsistent classes in tasks: %d in task 1, %d in task %d", ...
                length(unique(Ys{i})), length(unique(Ys{1})), i)
        end
    end
    
    n_task = length(Ys);
    labelSet = unique(Ys{1});
    W = cell(length(labelSet),1);
    c = cell(length(labelSet),1);
    Xs_ = cell(size(Xs));
    
    for i = 1:length(labelSet)
        Ys_ = Ys;
        for t = 1:n_task
            Xs_{t} = Xs{t}';
            Ys_{t} = double(Ys{t} == labelSet(i));
            Ys_{t}(Ys_{t}==0) = -1; %change from 0 to -1 for MALSAR settings
        end
        c{i} = zeros(1, n_task);
        W{i} = Least_NCFGLassoF2(Xs_, Ys_, rhos(1), rhos(2), opts);
    end
    
    OvRmod = struct('W', W, 'c', c, 'labelSet', labelSet);
end