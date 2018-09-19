from TPC_Configuration import *

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from scipy import odr
import pandas as pd
import optunity
import optunity.metrics
import os
from time import time, strftime, gmtime
from sklearn import preprocessing
import logging

## Everything to run Preprocessing by itself ##
###############################################
# start_time = strftime("%m_%d_%H:%M:%S", gmtime())
# Load hdf5 and pull list of Events ##############
# df_name = '/home/kelly/PycharmProjects/EDM_Support/Kr83m_s2_Areas_20Runs_10_20.h5'
# events_tree = pd.read_hdf(df_name, key='Kr83m')

# events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn < -18, :]
# events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn > -19, :]
# print(len(events_tree))
#
# events_tree = events_tree.loc[lambda df: df.s2_a > 8000, :]
# events_tree = events_tree.loc[lambda df: df.s2_a < 30000, :]

# dir_path = os.getcwd()
# plot_path = dir_path+'/EDM_Support/'+start_time+'/'
# if not os.path.exists(plot_path):
#     os.mkdir(plot_path)
# LOG_FILENAME = plot_path+'LogFile.log'
# logging.basicConfig(filename=LOG_FILENAME, level =logging.INFO)
# logging.warning('Abandon All Hope. There is no joy to be had here.')
# logging.info('Initial Dataframe: '+df_name)
# logging.info('Start Time: '+start_time)



single_dead = [[11, 13, 45,46,47, 12], [33, 35, 64, 34], [31, 61, 63, 87, 62], [35, 64, 36, 89, 65],
               [44, 45, 72, 74, 95, 96, 73], [52, 53, 78, 80, 99, 100, 79], [61, 85, 87, 105, 86],
               [63, 64, 87, 89, 106, 107, 88], [67, 68, 90, 92, 108, 109, 91], [82, 101, 103, 116, 102],
               [105, 117, 119, 125, 118]]  #Dead single PMTS

single_dead_comp = np.array([[5,7,40, 41,42, 6], [8, 10, 43, 9], [6, 40, 42, 70, 41], [6, 40, 42, 70, 41],
                      [47,48, 74, 76, 96, 97, 75], [47,48, 74, 76, 96, 97, 75], [41, 69, 71, 93, 70],
                      [48, 49, 75, 77, 97, 98,76], [71, 72, 93, 95, 110, 111, 94], [70, 92, 94, 110, 93],
                      [93, 109, 111, 121, 110]])  ## Handselected regions for pulling dead pmt estimation calibration

single_dead_comp_check = np.array([[17,19,50,51,52,18], [14,16,48,15], [24,55,57,82,56], [24,55,57,82,56],
                      [42,43,70,72,93,94,71], [42,43,70,72,93,94,71], [36,89,67,90,66],
                      [58,59,83,85,104,103,84], [76,77,97,99,113,114,98], [74,95,97,112,96],
                      [96,111,113,122,112]])  ## Handselected regions for pulling dead pmt estimation

double_dead = np.array([[0, 3, 36, 37, 38, 1], [0, 3, 36, 37, 38,  2],
                        [25, 28, 57, 58, 59, 26], [25, 28, 57, 58, 59, 27]])
double_dead_comp = np.array([[6, 9, 41, 42, 43, 7], [6, 9, 41, 42, 43, 8],
                             [6, 9, 41, 42, 43, 7], [6, 9, 41, 42, 43, 8]])
double_dead_comp_check = np.array([[18, 21, 51, 52, 53, 19], [12, 21, 51, 52, 53, 20],
                                   [18, 21, 51, 52, 53, 19], [12, 21, 51, 52, 53, 20]])

def dead_single_opt(pmts,pmts_check, events):
    N = int(len(events)/5)
    Events = [[event[j] for event in events if event[pmts[-1]] > 50][0:N] for j in pmts]
    logging.info('Number of Events Trained: '+str(len(Events[0])))
    logging.info('PMT Used to Train: '+str(pmts[-1]))
    data_train = list(zip(*Events[0:-1]))
    target_train = Events[-1]

    print('Normalizing Data')
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)

    # we explicitly generate the outer_cv decorator so we can use it twice
    outer_cv = optunity.cross_validated(x=data_train, y=target_train, num_folds=2)
    mse_old = 10e7
    def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
        """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""
        global optimal_parameters, clf
        # define objective function for tuning
        @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=2)
        def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
            # sample_weights = my_scaling_odr(y_train)
            # sample_weights = [i / max(Events[-1]) for i in Events[-1]]

            model = svm.SVR(C=C, gamma=gamma).fit(x_train, y_train)#, sample_weight=sample_weights
            predictions = model.predict(x_test)
            return optunity.metrics.mse(y_test, predictions)

        # optimize parameters
        optimal_pars, _, _ = optunity.minimize(tune_cv, 200, C=[1, 4000], gamma=[0, 10], pmap=optunity.pmap)
        logging.info("Optimal hyperparameters: " + str(optimal_pars))
        # sample_weights = my_scaling_odr(y_train)

        tuned_model = svm.SVR(**optimal_pars).fit(x_train, y_train)
        predictions = tuned_model.predict(x_test)
        mse = optunity.metrics.mse(y_test, predictions)
        logging.info('mse: ' + str(mse))
        if mse < mse_old:
            optimal_parameters = optimal_pars
            clf = tuned_model
        return mse


    # wrap with outer cross-validation
    compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)
    print('Beginning Cross-Validated Optimization of HyperParameters')
    compute_mse_rbf_tuned()
    Events_check = [[event[j] for event in events if event[pmts_check[-1]] > 50] for j in pmts_check]
    logging.info('Number of Events Trained: '+str(len(Events_check[0])))
    logging.info('PMT Used to Train Final Function: '+str(pmts_check[-1]))
    X_Span = list(zip(*Events_check[:-1]))
    X_Span = scaler.transform(X_Span)
    print('Predicting Data Now')
    pmt_estimate = clf.predict(X_Span)

    # print('Plotting Guessed Data Now')

    diff = [(pmt_estimate[i] - Events_check[-1][i]) / (Events_check[-1][i] + 1) for i in range(0, len(Events_check[-1]))]
    # print(np.mean(diff), np.std(diff))
    # print(np.mean(np.abs(diff)), np.std(np.abs(diff)))
    logging.critical('Final Average Absolute Relative Error: ' + str(round(np.mean(np.abs(diff)),3))
                     + '+-' +str(round(np.std(np.abs(diff)), 3)))

    # plt.figure()
    # plt.plot(Events_check[-1], pmt_estimate, '*')
    # plt.plot([0, max(Events_check[-1])], [0, max(Events_check[-1])], 'r', label='Error = 0%')
    # plt.xlabel('Actual PMT Value')
    # plt.ylabel('Estimated PMT Value')
    # plt.show()

    return clf, scaler


def exponential_func(params, x):
    return params[0]*np.exp(-1*params[1]*(x))

def my_scaling_odr(data):
    y, x = np.histogram(data, bins=int(np.sqrt(len(data))*3), range=[0, 400])
    x = [(x[i]+x[i+1])/2 for i in range(0, len(x)-1)]
    scaling_model = odr.Model(exponential_func)
    scaling_data = odr.RealData(x, y)
    scaling_odr = odr.ODR(scaling_data, scaling_model, beta0=[len(data), 20])
    output = scaling_odr.run()
    beta= output.beta
    red_chi = output.res_var
    sample_weights = [1/np.sqrt(exponential_func(beta, i)) for i in data]

    return sample_weights, red_chi


def fix_dead_pmts(dead_pmts, comp_pmts,comp_pmts_check, event_dataframe):
    start = time()
    logging.info('Operation Type: Preprocessing')
    events = event_dataframe['s2_area_array'].tolist()
    for i in range(0,len(dead_pmts)):
        logging.critical('Interpolating PMT #'+str(dead_pmts[i][-1]))
        dead_fit, scaler = dead_single_opt(comp_pmts[i], comp_pmts_check[i], events)
        logging.info('Finished Optimization and Fitting: '+str((time()-start)/60)+' min')
        Events = [[event[j] for event in events] for j in dead_pmts[i]]
        PMT_features = list(zip(*Events[:-1]))
        PMT_features = scaler.transform(PMT_features)
        pmt_estimate = dead_fit.predict(PMT_features)
        events = np.array(events)
        events[:, dead_pmts[i][-1]] = pmt_estimate
        events = events.tolist()
    return events


# better_pmt_array = fix_dead_pmts(double_dead,double_dead_comp, double_dead_comp_check,events_tree)
# events_tree['s2_area_array'] = better_pmt_array
# better_pmt_array = fix_dead_pmts(single_dead,single_dead_comp,single_dead_comp_check,events_tree)
# events_tree['s2_area_array'] = better_pmt_array
#
# events_tree.to_hdf(plot_path+'reincarnated_pmts.h5',key='df', mode='w')