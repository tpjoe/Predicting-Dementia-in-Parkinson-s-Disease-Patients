comb <- function(x, ...) {
  lapply(seq_along(x),
    function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}

crossvalidateOrdinal <- function(data, fixed, random, dependent, family, modelType='olmm'){
    # 'data' is the training set with the ".folds" column
    # 'k' is the number of folds we have
    # 'model' is a string describing a linear regression model formula
    # 'dependent' is a string with the name of the score column we want to predict
    # 'random' is a logical; do we have random effects in the model?

    # Initialize empty list for recording performances
    macroAUCs_train <- c()
    macroAUCs_test <- c()
    AUCs_0_test <- c()
    AUCs_1_test <- c()
    AUCs_2_test <- c()
    k <- max(as.integer(data$.folds))
    data <- as.data.frame(data)

    # One iteration per fold
    for (fold in 1:k){

        # Create training set for this iteration
        # Subset all the datapoints where .folds does not match the current fold
        training_set <- data[data$.folds != fold, ]

        # Create test set for this iteration
        # Subset all the datapoints where .folds matches the current fold
        testing_set <- data[data$.folds == fold, ]

        # Train linear mixed effects model on training set
        mod_olmm <- olmm(as.formula(paste(fixed, '+', random)), data = training_set, family = cumulative())

        # Predict the dependent variable in the testing_set with the trained model
        y_pred_train <- predict(mod_olmm, training_set, type = 'prob')
        y_pred_test <- predict(mod_olmm, testing_set, type = 'prob')
     
        # Get AUCs
        training_set[[dependent]] <- as.factor(as.character(training_set[[dependent]]))
        testing_set[[dependent]] <- as.factor(as.character(testing_set[[dependent]]))

        encoder_train <- onehot(as.data.frame(training_set[, dependent]))
        encoder_test <- onehot(as.data.frame(testing_set[, dependent]))
        dependent_encoded_train <- predict(encoder_train, as.data.frame(training_set[, dependent]))
        dependent_encoded_test <- predict(encoder_test, as.data.frame(testing_set[, dependent]))
        colnames(dependent_encoded_train) <- sapply(0:2, function(x) {paste0('y.Train', x, '_true')})
        colnames(dependent_encoded_test) <- sapply(0:2, function(x) {paste0('y.Test', x, '_true')})
     
        colnames(y_pred_train) <- sapply(0:2, function(x) {paste0('y.Train', x, '_pred_', modelType)})
        colnames(y_pred_test) <- sapply(0:2, function(x) {paste0('y.Test', x, '_pred_', modelType)})
        y_df_train <- cbind(dependent_encoded_train, y_pred_train)
        y_df_test <- cbind(dependent_encoded_test, y_pred_test)

        multiROC_train <- multi_roc(y_df_train)
        multiROC_test <- multi_roc(y_df_test)
        
        # Add the AUC to the performance list
        macroAUCs_train[fold] <- multiROC_train$AUC[[modelType]]$macro
        macroAUCs_test[fold] <- multiROC_test$AUC[[modelType]]$macro
        AUCs_0_test[fold] <- multiROC_test$AUC[[modelType]]$y.Test0
        AUCs_1_test[fold] <- multiROC_test$AUC[[modelType]]$y.Test1
        AUCs_2_test[fold] <- multiROC_test$AUC[[modelType]]$y.Test2
#         print(paste('Fold', k, 'done.', sep = ' '))
        }
    
    # Return the mean of the recorded RMSEs
    return(cbind('macroAUCs_train' = macroAUCs_train, 'macroAUCs_test' = macroAUCs_test,
                 'AUCs_0_test' = AUCs_0_test, 'AUCs_1_test' = AUCs_1_test, 'AUCs_2_test' = AUCs_2_test))
}