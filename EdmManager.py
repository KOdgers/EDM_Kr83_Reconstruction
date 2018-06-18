from SimilarityScaledPosRec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time, sleep


# Load hdf5 and pull list of Events ##############
events_tree = pd.read_hdf('/home/kelly/PycharmProjects/EDM_Support/Kr83m_s2_Areas_10Runs_10_20.h5', key='Kr83m')

events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn < -10, :]
events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn > -15, :]
print(len(events_tree))

events_tree = events_tree.loc[lambda df: df.s2_b < 15000, :]



event_list = events_tree['s2_area_array'].tolist()  # [['area','channel','int_a_x_3d_nn','y','z','s2_a','s2_b']]
event_x = events_tree['int_a_x_3d_nn'].tolist()
event_y = events_tree['int_a_y_3d_nn'].tolist()
Events = list(zip(event_list, event_x, event_y))
print(len(event_list))

tpc = TpcRep()
tpc.give_events(Events)

tpc.cut_worse_5_percent()
tpc.sklearn_mds()
# tpc.plot_edm_nn()
tpc.translation_scaling_rotation()
tpc.get_polar_errors()

sleep(1)
tpc.plot_edm_nn('cart')
tpc.plot_edm_nn('polar')
tpc.plot_edm_nn('polar_flipped')
# tpc.plot_edm_recon(cuts=True)

# distribution = tpc.get_distribution()
# nn_dist = tpc.get_nn_distribution()
# #
# time = strftime("%m_%d_%H:%M", gmtime())
# # Plotting Unscaled Distribution #########
#
#
# plt.figure(3, figsize=(4, 4))
# plt.plot(nn_dist[:, 0], nn_dist[:, 1], '*')
# plt.xlabel(' X-ish ')
# plt.ylabel(' Y-ish ')
# plt.title('Neural Net Position from Pax')
# plt.savefig('../EDM_Support/NN-Position'+time+'.png')


