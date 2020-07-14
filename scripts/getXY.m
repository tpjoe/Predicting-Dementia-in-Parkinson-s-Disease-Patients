function [Xs, Ys] = getXY(xydir, Xlead, Ylead, max_task)    
    % reading data from xydir
    X_list = dir(strcat(xydir, "/", Xlead, "*.*"));
    Y_list = dir(strcat(xydir, "/", Ylead, "*.*"));
    
    % For max task (last X)
    if max_task == "max"
        max_task = length(X_list);
    end
    
    X_list = X_list(1:max_task);
    Y_list = Y_list(1:max_task);

    Xs = cell(1, length(X_list));
    Ys = cell(1, length(Y_list));
    for i = 1:length(X_list)
        x = table2array(readtable(strcat(xydir, "/", X_list(i).name)));
        Xs{i} = x(:, 2:end);
        y = table2array(readtable(strcat(xydir, "/", Y_list(i).name)));
%         y(y==1) = 0;
%         y(y==0) = -1;
%         y(y==2) = 1;
        Ys{i} = y(:, 2:end);
    end 
end