from TPC_Configuration import ExcludedPMTS, active_pmts, qes_pmts, PMT_positions
import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime

from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle
import matplotlib
import os

class MC_EDM_Comp:

    def __init__(self):
        self.edm_distribution =[]
        self.nn_distribution =[]
        self.edm_patterns =[]
        self.flip = False

    def set_distributions(self, edm, nn, edm_patterns):
        self.edm_distribution = edm
        self.nn_distribution = nn
        self.edm_patterns= edm_patterns

    def plot_edm_nn(self, ptype):
        # Rescale the data
        X_nn = self.nn_distribution
        if ptype == 'cart':
            npos = self.edm_distribution
            npos *= np.sqrt((X_nn ** 2).sum()) / np.sqrt((npos ** 2).sum())
            the_title = 'Cartesian Plotting No Rotation'
        elif ptype == 'polar':
            npos = self.polar_dist
            npos = np.array([[npos[i, 0] * np.cos(npos[i, 1]),
                              npos[i, 0] * np.sin(npos[i, 1])] for i in range(len(npos))])
            the_title = 'Polar Distribution W/ Phi Correction'

        elif ptype == 'polar_flipped':
            npos = self.polar_dist_flipped
            npos = np.array([[npos[i, 0] * np.cos(npos[i, 1]),
                              npos[i, 0] * np.sin(npos[i, 1])] for i in range(len(npos))])
            the_title = 'Flipped Polar Distribution W/ Phi Correction'

        plt.figure(figsize=(6, 6))
        ax = plt.axes()
        s = 100
        plt.scatter(X_nn[:, 0], X_nn[:, 1], color='navy', s=s, lw=0,
                    label='NN Reconstructed')
        plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='MDS Reconstructed')
        plt.legend(scatterpoints=1, loc='best', shadow=False)

        # similarities = self.edm.max() / self.edm * 100
        # similarities[np.isinf(similarities)] = 0

        # Plot the edges
        segs = np.zeros((len(npos), 2, 2), float)
        segs[:, :, 1] = np.array([[npos[i, 1], X_nn[i, 1]] for i in range(len(npos))])
        segs[:, :, 0] = np.array([[npos[i, 0], X_nn[i, 0]] for i in range(len(npos))])

        # values = np.abs(similarities)
        lc = LineCollection(segs,
                            zorder=0, cmap=plt.cm.Blues)  # ,
        # norm=plt.Normalize(0, values.max()))
        # lc.set_array(similarities.flatten())
        lc.set_linewidths(2 * np.ones(len(segs)))
        ax.add_collection(lc)
        plt.title(the_title)
        plt.savefig('../EDM_Support/' + ptype + strftime("%m_%d_%H:%M:%S", gmtime()) + '.png')
        plt.close()

    def translation_scaling_rotation(self):
        X_nn = self.nn_distribution
        npos = self.edm_distribution
        self.polar_dist = np.zeros_like(self.edm_distribution)
        self.polar_dist_flipped = np.zeros_like(self.edm_distribution)
        self.polar_nn_dist = np.zeros_like(self.edm_distribution)

        npos *= np.sqrt((X_nn ** 2).sum()) / np.sqrt((npos ** 2).sum())
        x_dist = np.array([X_nn[i, 0] ** 2 + X_nn[i, 1] ** 2 for i in range(0, len(X_nn))])
        origin_index = np.argmin(x_dist)
        npos = np.array(
            [[npos[i, 0] - npos[origin_index, 0], npos[i, 1] - npos[origin_index, 1]] for i in range(len(npos))])
        for i in range(0, len(npos)):
            theta = np.abs(np.arctan(
                (npos[i, 1] - npos[origin_index, 1]) / (npos[i, 0] - npos[origin_index, 0])))
            theta2 = np.abs(np.arctan(
                (X_nn[i, 1] - X_nn[origin_index, 1]) / (X_nn[i, 0] - X_nn[origin_index, 0])))
            if npos[i, 0] > 0:
                if npos[i, 1] < 0:
                    theta = 2 * np.pi - theta
            elif npos[i, 0] <= 0:
                if npos[i, 1] > 0:
                    theta = np.pi - theta
                else:
                    theta = np.pi + theta

            if X_nn[i, 0] > 0:
                if X_nn[i, 1] < 0:
                    theta2 = 2 * np.pi - theta2

            else:
                if X_nn[i, 1] > 0:
                    theta2 = np.pi - theta2
                else:
                    theta2 = np.pi + theta2

            self.polar_dist[i, 0] = np.sqrt((npos[i, 0] - npos[origin_index, 0]) ** 2 +
                                            (npos[i, 1] - npos[origin_index, 1]) ** 2)
            self.polar_nn_dist[i, 0] = np.sqrt((X_nn[i, 0] - X_nn[origin_index, 0]) ** 2 +
                                               (X_nn[i, 1] - X_nn[origin_index, 1]) ** 2)
            self.polar_dist_flipped[i, 0] = self.polar_dist[i, 0]
            self.polar_dist[i, 1] = theta % (2 * np.pi)
            self.polar_nn_dist[i, 1] = theta2 % (2 * np.pi)
            self.polar_dist_flipped[i, 1] = 2 * np.pi - 1 * self.polar_dist[i, 1]
        x_dist = np.array([(50 - X_nn[i, 0]) ** 2 + X_nn[i, 1] ** 2 for i in range(0, len(X_nn))])
        second_index = np.argmin(x_dist)
        phi_edm = self.polar_dist[second_index, 1]
        phi_edm_flipped = self.polar_dist_flipped[second_index, 1]
        phi_nn = self.polar_nn_dist[second_index, 1]

        phi_shift = phi_nn - phi_edm
        phi_shift2 = phi_nn - phi_edm_flipped

        for i in range(len(npos)):
            self.polar_dist[i, 1] = self.polar_dist[i, 1] + phi_shift
            self.polar_dist_flipped[i, 1] = self.polar_dist_flipped[i, 1] + phi_shift2

    def corrections(self, cuts=True):
        self.translation_scaling_rotation()
        if cuts:
            count = 0
            for i in range(0, len(self.polar_dist)):
                if self.polar_dist[i, 0] > 50:
                    count += 1
                    self.nn_distribution=np.delete(self.nn_distribution,i-count,0)
                    self.edm_distribution=np.delete(self.edm_distribution,i-count,0)
                    self.edm_patterns = np.delete(self.edm_patterns,i-count,0)
        print(count)
        self.translation_scaling_rotation()



    def get_polar_errors(self):
        plt.figure(11, figsize=(9, 4))
        plt.title('Errors (Un-mirrored, Mirrored)')
        errors = [0, 0]
        for j in range(0, 2):
            edm_dist = self.polar_dist_flipped
            ptype = 'flipped'
            if j == 0:
                edm_dist = self.polar_dist
                ptype = 'not_flipped'
                print('yup')
            else:
                print('also yup')

            theta_err = np.zeros(len(edm_dist))
            rad_err = np.zeros(len(edm_dist))

            for i in range(len(edm_dist)):
                ang_err = np.pi - np.abs(np.abs(edm_dist[i, 1] - self.polar_nn_dist[i, 1]) - np.pi)
                direction = 1
                if (np.abs(edm_dist[i, 1] - self.polar_nn_dist[i, 1]) > (edm_dist[i, 1] - self.polar_nn_dist[i, 1]) and
                        np.abs(edm_dist[i, 1] - self.polar_nn_dist[i, 1]) > ang_err):
                    direction = -1

                theta_err[i] = direction * ang_err
                rad_err[i] = edm_dist[i, 0] - self.polar_nn_dist[i, 0]

            plt.subplot(1, 2, j + 1)
            plt.xlabel('Difference in Radii [cm]')
            plt.ylabel('Error in Angle [rad]')
            plt.plot(rad_err, theta_err, '.')
            errors[j] = np.sum(np.abs(rad_err))

        plt.xlabel('Difference in Radii [cm]')
        plt.ylabel('Error in Angle [rad]')
        plt.savefig('../EDM_Support/Errs' + strftime("%m_%d_%H:%M:%S", gmtime()) + '.png')
        plt.close()
        if errors[1]<errors[0]:
            self.flip = True
        return

    def make_pmt_position_maps(self):
        if self.flip:
            X_edm = self.polar_dist_flipped
        else:
            X_edm = self.polar_dist
        directory = '../EDM_Support/' + strftime("%m_%d_%H:%M:%S", gmtime()) + '/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        for i in range(0, len(self.edm_patterns)):
            self.pmt_pattern_map(i, X_edm[i, :], directory)

    def pmt_pattern_map(self, pattern_number, position, directory):
        Counts_Per_PMT = self.edm_patterns[pattern_number].get_pattern()
        top_channels = list(range(0, 127))
        bottom_channels = list(range(127, 247 + 1))
        PMT_distance_top = 7.95  # cm
        PMT_distance_bottom = 8.0  # cm
        PMTOuterRingRadius = 3.875  # cm
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(Counts_Per_PMT[2:126]), clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)

        fig = plt.figure(figsize=(9, 8))
        #     fig.suptitle('Coincidence for PMT '+str(MainPMT))
        ax1 = fig.add_axes([0.9, 0.0, 0.1, 1])
        ax2 = fig.add_axes([0., 0.0, 0.9, 1])

        # PMT QEs
        name = "PAX_PMTpattern"
        plot_radius = 60
        axes = plt.gca()
        #     fig = plt.figure(figsize=(15,15))

        # plt.subplot(121)
        ax2.set_xlim((-plot_radius, plot_radius))
        ax2.set_ylim((-plot_radius, plot_radius))

        patches = []
        for ch in top_channels:
            if ch in ExcludedPMTS:
                circle = Circle((PMT_positions[ch]['x'], PMT_positions[ch]['y']), PMTOuterRingRadius, color='k')
            else:
                circle = Circle((PMT_positions[ch]['x'], PMT_positions[ch]['y']), PMTOuterRingRadius,
                                facecolor=mapper.to_rgba(Counts_Per_PMT[ch]))

            patches.append(circle)
            ax2.annotate(str(ch), xy=(PMT_positions[ch]['x'], PMT_positions[ch]['y']), fontsize=14, ha='center',
                         va='center')
        square = Rectangle((position[0] * np.cos(position[1]), position[0] * np.sin(position[1])), PMTOuterRingRadius, PMTOuterRingRadius,
                        color='r')
        patches.append(square)
        p = PatchCollection(patches, alpha=1.0, match_original=True)
        ax2.add_collection(p)
        ax2.text(0.08, 1.05, 'PMT Pattern #'+str(pattern_number), transform=axes.transAxes, horizontalalignment='left',
                 verticalalignment='top', fontsize=16)

        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=matplotlib.cm.jet,
                                               norm=norm,
                                               orientation='vertical')
        ax1.yaxis.set_label_position("right")
        ax1.set_ylabel('Number of Counts', fontsize=14)
        # plt.plot(position[0]*np.cos(position[1]), position[0]*np.sin(position[1]),'k*')
        plt.savefig(directory+'Pattern'+str(pattern_number)+'.png')
        plt.close()
