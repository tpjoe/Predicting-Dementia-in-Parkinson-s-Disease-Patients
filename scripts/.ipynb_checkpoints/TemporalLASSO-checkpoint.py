# Import lib
import matlab.engine
import pickle
from sklearn import preprocessing 
import pandas as pd
import numpy as np

import random
# from ipypb import track
from time import sleep
import numpy as np

from pymongo import MongoClient
import datetime 
import os
import pickle 
import pandas as pd


def main(model_type):
    # which XY to use
    eng.workspace['Xlead'] = "X_yearsince1stvisit_stdScaled." #"X_yearsince1stvisit", "X_visit", "X_yearsince1stvisit_stdScaled_log10.", "X_yearsince1stvisit_stdScaled."
    eng.workspace['Ylead'] = "y_yearsince1stvisit_stdScaled." #"y_yearsince1stvisit", "y_visit", "y_yearsince1stvisit_stdScaled_log10.", "y_yearsince1stvisit_stdScaled."

    # Model params
    classes = ['no CI', 'CI', 'dementia']
    eng.workspace['iter'] = 100
    eng.workspace['cv_method'] = "stratified sampling" # cv or weighted sampling or cv2 or stratified sampling
    eng.workspace['cvfold'] = 10.0 #have to be DECIMAL here (sometimes 5 cause error cos latter years don't have enough)
    eng.workspace['test_frac'] = 0.25
    eng.workspace['outer_cv_method'] = "stratified holdout" #stratified holdout or capped weighted sampling
    eng.workspace['max_task'] = "max"

    if model_type == "TGL":
        Rho1 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        Rho2 = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
        Rho3 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    elif model_type == "CFGL":
        Rho1 = [0.00025, 0.005, 0.01]
        Rho2 = [0.005, 0.025, 0.05]
        Rho3 = [0.00025, 0.005, 0.01]
    elif model_type =="nFSGL":
        Rho1 = [0.005, 0.01, 0.05, 0.1]
        Rho2 = [0.005, 0.01, 0.05, 0.1]
        Rho3 = [0]


    ############################## AUTOMATED BELOW HERE ##############################

    eng.workspace['model_type'] = model_type
    eng.workspace['Rho1'] = matlab.double(Rho1)
    eng.workspace['Rho2'] = matlab.double(Rho2)
    eng.workspace['Rho3'] = matlab.double(Rho3)

    # Run model
    eng.main_ordinal(nargout=0)

    # Rearrange Y_pred and Y_test to dictionary format
    y_pred_mat = np.array(eng.workspace['Ys_pred_mat'])
    y_test_mat = np.array(eng.workspace['Ys_test_mat'])
    n_class = len(classes)
    predictTask = ''.join(str(n) for n in list(np.array(range(int(eng.eval('length(Ys)')))) + 1)) #ex. '123' 

    y_pred_dict = {class_key: {('task' + str(task_key)): [] for task_key in range(1, (int(predictTask[-1]) + 1))}
                   for class_key in classes}
    y_test_dict = {('task' + str(task_key)): [] for task_key in range(1, (int(predictTask[-1]) + 1))}
    n_in_task = np.array(eng.workspace['test_size'])
    n_index = np.insert(np.cumsum(n_in_task), obj=0, values=0)

    for i_task, task in enumerate(y_pred_dict[classes[0]].keys()):
        selected_rows = range(int(n_index[int(i_task)]), int(n_index[int(i_task+1)]))
        y_test_dict[task] = y_test_mat[selected_rows[0]:selected_rows[-1], :].tolist()
        for i_class, class_ in enumerate(y_pred_dict.keys()):
            selected_columns = np.arange(i_class, y_pred_mat.shape[1], n_class)
    #         print(selected_columns)
    #         print(selected_rows)
            y_pred_dict[class_][task] = y_pred_mat[selected_rows[0]:selected_rows[-1], selected_columns].tolist()

    ############################## AUTOMATED ABOVE HERE ##############################

    # Saving to mongodb
    import json

    print('startedMongo')
    client = MongoClient()
    db = client.udall2Results
    multitaskModels = db.multitaskModels
    date = datetime.datetime.now()
    result_data = {
        'exp': multitaskModels.count_documents({}) + 1,
        'iterations': eng.workspace['iter'],
        'run type': 'test',
        'description': 'Task# = # years since 1st visit',  ###<-------------
        'task criteria': '1' + '-' + str(int(eng.eval('length(Xs)'))),
        'model type': 'ordinal',
        'date': [date.month, date.day, date.year],
        'date time': date,
        'train': 'task1',
        'predict': 'task' + predictTask,
        'test fraction': eng.workspace['test_frac'],
        'stratify': True,
        'preprocessing': 'one-hot encoded/std scaling', ###<------------- "minmax scaling", "std scaling"
        'X columns': None,
        'X': None,
        'y': None,
        'y_test': y_test_dict,
        'y_pred': y_pred_dict, 
        'clfs': model_type,
        'CVparams': {'Rho1': Rho1, 'Rho2': Rho2, 'Rho3': Rho3},
        'grid CV': eng.workspace['cvfold'],
        'cv_method': eng.workspace['cv_method'],
        'outer_cv_method': eng.workspace['outer_cv_method'],
        'scores: macro AUC': np.array(eng.workspace['macro_AUC']).tolist(),
        'scores: micro AUC': None,
        'scores: individual AUC': np.array(eng.workspace['AUCs']).tolist(),
        'elasticnet weight (mean)': np.array(eng.workspace['W_mean']).tolist(),
        'elasticnet weight (median)': None,
        'feature importance (mean)': None,
        'feature importance (median)': None,
        'hyperparams':  [np.array(eng.workspace['param_Rho1']).tolist(),
                         np.array(eng.workspace['param_Rho2']).tolist(),
                         np.array(eng.workspace['param_Rho3']).tolist()]
    }
    result = multitaskModels.insert_one(result_data)
    print('finishedMongo')
    return





# Start the matlab session
eng = matlab.engine.start_matlab()
print('Matlab engine initiated.')
eng.eval("distcomp.feature('LocalUseMpiexec', false)", nargout=0)
eng.eval("poolObj = parpool(36)", nargout=0)
print('Parpool completed.')
eng.cd(r'~/tpjoe@stanford.edu/project_Udall2/UDALL2_project/scripts')

for model_type in ["TGL"]:
    main(model_type)
eng.eval('delete(poolObj)', nargout=0)
eng.quit()