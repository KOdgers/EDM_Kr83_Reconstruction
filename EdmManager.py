from SimilarityScaledPosRec import *
from EDM_Dev_Analysis import *
from DimensionReduction import *
from Preprocessing_PMT_Patterns import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

start = time.time()
# Load hdf5 and pull list of Events ##############
# events_tree = pd.read_hdf('/home/kelly/PycharmProjects/EDM_Support/Kr83m_s2_Areas_20Runs_10_20.h5', key='Kr83m')
events_tree = pd.read_hdf('/home/kelly/PycharmProjects/EDM_Kr83_Reconstruction/EDM_Support/09_18_17:55:43/reincarnated_pmts.h5', key='df')

dir_path = os.getcwd()
plot_path = dir_path+'/EDM_Support/'
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

# Data Selection

events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn < -18, :]
events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn > -19, :]
print(len(events_tree))

events_tree = events_tree.loc[lambda df: df.s2_a > 8000, :]
events_tree = events_tree.loc[lambda df: df.s2_a < 30000, :]

# Preprocessing to reconstruct dead PMTS - Only necessary if you are using a raw dataframe and want to fill the dead pmts
# better_pmt_array = fix_dead_pmts(dead_pmts,comp_pmts, comp_pmts_check,events_tree)
# events_tree['s2_area_array'] = better_pmt_array
# events_tree.to_hdf(plot_path+'reincarnated_pmts.h5',key='df', mode='w')



# List Formatting
event_list = events_tree['s2_area_array'].tolist()  # [['area','channel','int_a_x_3d_nn','y','z','s2_a','s2_b']]
event_x = events_tree['int_a_x_3d_nn'].tolist()
event_y = events_tree['int_a_y_3d_nn'].tolist()
Events = list(zip(event_list, event_x, event_y))
print(len(event_list))
del events_tree, event_list, event_x, event_y
weights = np.ones([127])
# weights = weighting_calc(20)



## Create EDM ##
############################################
print ('Begining EDM Construction')
tpc = TpcRep(plot_path)
tpc.give_events(Events, 4000, weights)
tpc.cut_worse_5_percent()

# Apply Dimensionality Reduction. Intrinsic dimension = ~2 #
############################################################
print('Beginning Dimensional Reduction')
manifold = Reduction(plot_path)
manifold.set_edm(tpc.edm)
# manifold.load_edm('08_21_19:26:22')
manifold.sklearn_mds()
# manifold.sklearn_local_linear(50)

times = manifold.save_edm()
manifold.save_edm_distribution()

# Analysis of Reconstructed Distribution #
##########################################
print('Cuts and Orienting')
posrec_analysis = MC_EDM_Comp(plot_path)
posrec_analysis.set_distributions(manifold.get_distribution(), tpc.get_nn_distribution(), tpc.get_patterns())
times = posrec_analysis.save_distributions(times)
posrec_analysis.load_distributions(times)
posrec_analysis.corrections()

print('Errors and Plotting')
posrec_analysis.get_polar_errors()
posrec_analysis.get_edm_cart()
mean_err, std_err = posrec_analysis.get_cart_error()
print(mean_err, std_err)

# Visualization #
#################

time.sleep(1)
posrec_analysis.plot_edm_nn('polar')
posrec_analysis.plot_edm_nn('polar_flipped')
posrec_analysis.get_radial_dist()
posrec_analysis.plot_edm()
#
#
# # Now With LLE #
# ################
#
# print('Beginning Second Reduction. LLE')
# manifold.sklearn_local_linear(20)
# times = manifold.save_edm()
# manifold.save_edm_distribution()
#
# # Analysis of Reconstructed Distribution #
# ##########################################
#
# print('Cuts and Orienting')
# posrec_analysis = MC_EDM_Comp(plot_path)
# posrec_analysis.set_distributions(manifold.get_distribution(), tpc.get_nn_distribution(), tpc.get_patterns())
# times = posrec_analysis.save_distributions(times)
# posrec_analysis.load_distributions(times)
# posrec_analysis.corrections()
#
# print('Errors and Plotting')
# posrec_analysis.get_polar_errors()
# posrec_analysis.get_edm_cart()
# mean_err, std_err = posrec_analysis.get_cart_error()
# print(mean_err, std_err)
#
# # Visualization #
# #################
#
# time.sleep(1)
# posrec_analysis.plot_edm_nn('polar')
# posrec_analysis.plot_edm_nn('polar_flipped')
# posrec_analysis.get_radial_dist()
# posrec_analysis.plot_edm()




print('Took '+ str(((time.time()-start)/60)) + ' minutes to complete')


# tpc.sklearn_local_linear()
# posrec_analysis = MC_EDM_Comp(plot_path)
# posrec_analysis.set_distributions(tpc.get_distribution(), tpc.get_nn_distribution(), tpc.get_patterns())
#
# posrec_analysis.corrections()
# posrec_analysis.get_polar_errors()
# posrec_analysis.get_edm_cart()
# mean_err, std_err = posrec_analysis.get_cart_error()
# print(mean_err, std_err)
# sleep(1)
# posrec_analysis.plot_edm_nn('polar')
# posrec_analysis.plot_edm_nn('polar_flipped')
# posrec_analysis.make_pmt_position_maps()



