from SimilarityScaledPosRec import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import gmtime, strftime


# Load hdf5 and pull list of Events ##############
events_tree = pd.read_hdf('/home/kelly/Data/XenonData/Kr83m_s2_area__10Runs_5_15.h5', key='Kr83m')

events_tree = events_tree.loc[lambda df: df.int_a_z_3d_nn > -8, :]


event_list = events_tree['Areas'].tolist()  # [['area','channel','int_a_x_3d_nn','y','z','s2_a','s2_b']]
event_x = events_tree['int_a_x_3d_nn'].tolist()
event_y = events_tree['int_a_y_3d_nn'].tolist()
Events = list(zip(event_list,event_x,event_y))
print(len(event_list))

tpc = TpcRep()
tpc.give_events(Events)
tpc.mds_classic()

distribution = tpc.get_distribution()
x_nn, y_nn = tpc.get_nn_distribution()

time = strftime("%m_%d_%H:%M", gmtime())
# Plotting Unscaled Distribution #########
plt.figure(1)
plt.subplot(121)
plt.plot(distribution[:, 0], distribution[:, 1], '*')
plt.xlabel(' X-ish ')
plt.ylabel(' Y-ish ')
plt.title('Unscaled Distribution after Classical MDS')

plt.subplot(122)
plt.plot(distribution[:, 0], distribution[:, 1], '*')
plt.xlabel(' X-ish ')
plt.ylabel(' Y-ish ')
plt.title('Unscaled Distribution after Classical MDS')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.savefig('../EDM_Support/FuckItLetsGo'+time+'.png')

plt.figure(2)
plt.plot(x_nn, y_nn,'*')
plt.xlabel(' X-ish ')
plt.ylabel(' Y-ish ')
plt.title('Neural Net Position from Pax')
plt.savefig('../EDM_Support/NN-Position'+time+'.png')


