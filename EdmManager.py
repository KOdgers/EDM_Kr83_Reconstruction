from SimilarityScaledPosRec import *
from EDM_Dev_Analysis import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time, sleep


# Load hdf5 and pull list of Events ##############
events_tree = pd.read_hdf('/home/kelly/PycharmProjects/EDM_Support/Kr83m_s2_Areas_10Runs_10_20.h5', key='Kr83m')

events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn < -10, :]
events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn > -12, :]
print(len(events_tree))

events_tree = events_tree.loc[lambda df: df.s2_b < 1000, :]



event_list = events_tree['s2_area_array'].tolist()  # [['area','channel','int_a_x_3d_nn','y','z','s2_a','s2_b']]
event_x = events_tree['int_a_x_3d_nn'].tolist()
event_y = events_tree['int_a_y_3d_nn'].tolist()
Events = list(zip(event_list, event_x, event_y))
print(len(event_list))

tpc = TpcRep()
tpc.give_events(Events, 400)

tpc.cut_worse_5_percent()
tpc.sklearn_mds()
# posrec_analysis = MC_EDM_Comp()
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

#
# tpc.sklearn_local_linear()
posrec_analysis = MC_EDM_Comp()
posrec_analysis.set_distributions(tpc.get_distribution(), tpc.get_nn_distribution(), tpc.get_patterns())

posrec_analysis.corrections()
posrec_analysis.get_polar_errors()
posrec_analysis.get_edm_cart()
mean_err, std_err = posrec_analysis.get_cart_error()
print(mean_err, std_err)
sleep(1)
posrec_analysis.plot_edm_nn('polar')
posrec_analysis.plot_edm_nn('polar_flipped')
# posrec_analysis.make_pmt_position_maps()



