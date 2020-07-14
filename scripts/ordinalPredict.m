function [Ys_pred] = ordinalPredict(Xs, OrdinalMod)

    % OrdinalMod is an object returned from cv_OvRTemporalLASSO
    % Xs is a cell with each column representing each task
    % Returning Ys_pred which is a cell with {1 x task} dimension with each
    % cell having {n_sample, n_class} dimension.

    labelSet = OrdinalMod(1).labelSet;
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
      for class = [1, 3, 2]
        % predict the prob of Target == 0 from weights of Target > 0
        if class == 1
            prob = Xs{t} * OrdinalMod(class).W(:, t) + OrdinalMod(class).c(t);
            y_pred = (1./(1 + exp(1).^(-1*prob)));
            y_pred = 1 - y_pred;
            Ys_pred{t}(:, class) = y_pred;
        end
        
        % predict the prob of Target == 2 from weights of Target > 1
        if class == 3
            prob = Xs{t} * OrdinalMod(class - 1).W(:, t) + OrdinalMod(class - 1).c(t);
            y_pred = (1./(1 + exp(1).^(-1*prob)));
            Ys_pred{t}(:, class) = y_pred;
        end
        
        % predict the prob of Target == 1 from prediction subtraction
        if class == 2
            Ys_pred{t}(:, class) = (1 - Ys_pred{t}(:, 1)) - Ys_pred{t}(:, 3);
        end
      end
%       col_sum = sum(Ys_pred{t}, 2);
%       Ys_pred{t} = bsxfun(@rdivide, Ys_pred{t}, col_sum);
    end
end
