function [Ys_pred] = OvRPredict(Xs, OvRmod)

    % OvRmod is an object returned from cv_OvRTemporalLASSO
    % Xs is a cell with each column representing each task
    % Returning Ys_pred which is a cell with {1 x task} dimension with each
    % cell having {n_sample, n_class} dimension.

    labelSet = OvRmod(1).labelSet;
    n_class = length(labelSet);
    n_task = length(Xs);
    Ys_pred = cell(1, n_task);
        
    for t = 1:n_task
        [m, n] =  size(Xs{t});
        if n ~= 58
            warning('Something could be wrong!!')
        end
        Ys_pred{1, t} = zeros(m, n_class);
    end
    
    for t = 1:n_task  
      for class = 1:n_class 
        prob = Xs{t} * OvRmod(class).W(:, t) + OvRmod(class).c(t);
        y_pred = (1./(1 + exp(1).^(-1*prob)));
        Ys_pred{t}(:, class) = y_pred;
      end
      col_sum = sum(Ys_pred{t}, 2);
      Ys_pred{t} = bsxfun(@rdivide, Ys_pred{t}, col_sum);
    end
end