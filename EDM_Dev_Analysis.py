from TPC_Configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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

    def translation_scaling_rotation_nn(self):
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
        self.pmt_pattern_map(origin_index, self.polar_dist[origin_index, :], self.polar_nn_dist[origin_index, :], '../EDM_Support/')
        self.pmt_pattern_map(second_index, self.polar_dist[second_index, :], self.polar_nn_dist[second_index, :], '../EDM_Support/')


    def tsr(self):
        """
        Function finds the best two patterns to use to set origin and horizontal axis
        It then generates two transformations that center/scale/flip/rotate the original configuration according to
        pmt array and tpc geometry
        :return:
        """
        X_nn = self.nn_distribution
        npos = self.edm_distribution
        self.polar_dist = np.zeros_like(self.edm_distribution)
        self.polar_dist_flipped = np.zeros_like(self.edm_distribution)
        self.polar_nn_dist = np.zeros_like(self.edm_distribution)

        npos *= np.sqrt((X_nn ** 2).sum()) / np.sqrt((npos ** 2).sum())
        # x_dist = np.array([X_nn[i, 0] ** 2 + X_nn[i, 1] ** 2 for i in range(0, len(X_nn))])
        # origin_index, score, coeffs_0 = self.find_best_pattern_xy(0, 0)
        # positions = npos[origin_index,:]
        x_shift = np.mean(npos[:, 0])
        y_shift = np.mean(npos[:, 1])

        # npos = np.array(
        #     [[npos[i, 0] - npos[origin_index, 0], npos[i, 1] - npos[origin_index, 1]] for i in range(len(npos))])
        npos = np.array(
            [[npos[i, 0] - x_shift, npos[i, 1] - y_shift] for i in range(len(npos))])
        for i in range(0, len(npos)):
            # theta = np.abs(np.arctan(
            #     (npos[i, 1] - npos[origin_index, 1]) / (npos[i, 0] - npos[origin_index, 0])))
            theta = np.abs(np.arctan(
                (npos[i, 1]) / (npos[i, 0])))
            if npos[i, 0] == 0:
                theta = np.pi/2

            theta2 = np.abs(np.arctan(
                (X_nn[i, 1]) / (X_nn[i, 0])))
            if X_nn[i,0] == 0:
                theta2 = np.pi/2
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

            # self.polar_dist[i, 0] = np.sqrt((npos[i, 0] - npos[origin_index, 0]) ** 2 +
            #                                 (npos[i, 1] - npos[origin_index, 1]) ** 2)
            # self.polar_nn_dist[i, 0] = np.sqrt((X_nn[i, 0] - X_nn[origin_index, 0]) ** 2 +
            #                                    (X_nn[i, 1] - X_nn[origin_index, 1]) ** 2)
            # self.polar_dist_flipped[i, 0] = self.polar_dist[i, 0]
            # self.polar_dist[i, 1] = theta % (2 * np.pi)
            # self.polar_nn_dist[i, 1] = theta2 % (2 * np.pi)
            # self.polar_dist_flipped[i, 1] = 2 * np.pi - 1 * self.polar_dist[i, 1]
            self.polar_dist[i, 0] = np.sqrt((npos[i, 0]) ** 2 +
                                            (npos[i, 1]) ** 2)
            self.polar_nn_dist[i, 0] = np.sqrt((X_nn[i, 0]) ** 2 +
                                               (X_nn[i, 1]) ** 2)
            self.polar_dist_flipped[i, 0] = self.polar_dist[i, 0]
            self.polar_dist[i, 1] = theta % (2 * np.pi)
            self.polar_nn_dist[i, 1] = theta2 % (2 * np.pi)
            self.polar_dist_flipped[i, 1] = 2 * np.pi - 1 * self.polar_dist[i, 1]
        # x_dist = np.array([(50 - X_nn[i, 0]) ** 2 + X_nn[i, 1] ** 2 for i in range(0, len(X_nn))])
        second_index, score, x_2, y_2 = self.find_best_pattern_xy()
        theta_best = np.abs(np.arctan(y_2 / x_2))
        if x_2 > 0:
            if y_2 < 0:
                theta_best = 2 * np.pi - theta_best
        else:
            if y_2 > 0:
                theta_best = np.pi - theta_best
            else:
                theta_best = np.pi + theta_best

        phi_edm = self.polar_dist[second_index, 1] - theta_best
        phi_edm_flipped = self.polar_dist_flipped[second_index, 1] - theta_best
        phi_nn = self.polar_nn_dist[second_index, 1]
        positions = npos[second_index, :]


        for i in range(len(npos)):
            self.polar_dist[i, 1] = self.polar_dist[i, 1] - phi_edm
            self.polar_dist_flipped[i, 1] = self.polar_dist_flipped[i, 1] - phi_edm_flipped
        # self.pmt_pattern_map(origin_index, self.polar_dist[origin_index, :], self.polar_nn_dist[origin_index, :], '../EDM_Support/')
        self.pmt_pattern_map(second_index, self.polar_dist[second_index, :], self.polar_nn_dist[second_index, :], '../EDM_Support/','')
        self.pmt_pattern_map(second_index, self.polar_dist_flipped[second_index, :], self.polar_nn_dist[second_index, :], '../EDM_Support/','Flipped')

        self.edm_distribution = npos

    def corrections(self, cuts=True):
        self.tsr()
        # self.translation_scaling_rotation_nn()
        if cuts:
            count = 0
            for i in range(0, len(self.polar_dist)):
                if self.polar_dist[i, 0] > 50:

                    self.nn_distribution=np.delete(self.nn_distribution, i-count, 0)
                    self.edm_distribution=np.delete(self.edm_distribution, i-count, 0)
                    self.edm_patterns = np.delete(self.edm_patterns, i-count, 0)
                    count += 1
            print(count)
            self.tsr()



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
        X_nn = self.polar_nn_dist
        directory = '../EDM_Support/' + strftime("%m_%d_%H:%M:%S", gmtime()) + '/'
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        for i in range(0, len(self.edm_patterns)):
            self.pmt_pattern_map(i, X_edm[i, :],X_nn[i, :], directory)

    def pmt_pattern_map(self, pattern_number, position, nn_position, directory,flipped):
        Counts_Per_PMT = self.edm_patterns[pattern_number].get_pattern()
        top_channels = list(range(0, 127))
        bottom_channels = list(range(127, 247 + 1))
        PMT_distance_top = 7.95  # cm
        PMT_distance_bottom = 8.0  # cm
        PMTOuterRingRadius = 3.875  # cm
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max(Counts_Per_PMT[2:126]), clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.jet)

        fig = plt.figure(figsize=(9, 8))

        ax1 = fig.add_axes([0.9, 0.0, 0.1, 1])
        ax2 = fig.add_axes([0., 0.0, 0.9, 1])


        plot_radius = 60
        axes = plt.gca()

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
        square = Rectangle((position[0] * np.cos(position[1]), position[0] * np.sin(position[1])), PMTOuterRingRadius,
                           PMTOuterRingRadius, color='r')
        square2 = Rectangle((nn_position[0] * np.cos(nn_position[1]), nn_position[0] * np.sin(nn_position[1])),
                            PMTOuterRingRadius, PMTOuterRingRadius, color='b')
        patches.append(square)
        patches.append(square2)
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
        plt.savefig(directory+'Pattern'+flipped+str(pattern_number)+'.png')
        plt.close()


    def get_gaussian(self, pattern_number):

        pattern = self.edm_patterns[pattern_number].get_pattern()
        bins = np.linspace(-50, 50, 100)
        bin_mids = [(bins[i]+bins[i+1])/2 for i in range(0,len(bins)-1)]
        x_weights =[]
        y_weights =[]
        for i in range(0, len(bins)-1):
            x_weights.append(0+sum([pattern[j] for j in active_pmts if
                                   bins[i] < PMT_positions[j]['x'] < bins[i+1]])/
                                   (1+ len([pattern[j] for j in active_pmts if
                                   bins[i] < PMT_positions[j]['x'] < bins[i+1]])))
            y_weights.append(0+sum([pattern[j] for j in active_pmts if
                                   bins[i] < PMT_positions[j]['y'] < bins[i+1]])/
                                   (1+len([pattern[j] for j in active_pmts if
                                   bins[i] < PMT_positions[j]['y'] < bins[i+1]])))

        p0 = [1., 0., 1.]
        try:
            x_coefficients, x_var_matrix = curve_fit(Gaussian, bin_mids, x_weights, p0=p0)
            residuals = x_weights - Gaussian(bin_mids, x_coefficients[0], x_coefficients[1], x_coefficients[2])
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((x_weights - np.mean(x_weights)) ** 2)
            x_r_squared = 1 - (ss_res / ss_tot)
        except:
            return 0, 0, 0, 0, 0, 0

        try:
            y_coefficients, y_var_matrix = curve_fit(Gaussian, bin_mids, y_weights, p0=p0)
            residuals = y_weights - Gaussian(bin_mids, y_coefficients[0], y_coefficients[1], y_coefficients[2])
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_weights - np.mean(y_weights)) ** 2)
            y_r_squared = 1 - (ss_res / ss_tot)
        except:
            return 0, 0, 0, 0, 0, 0
        return x_coefficients[1], y_coefficients[1], x_coefficients[2], y_coefficients[2], x_r_squared, y_r_squared

    def find_best_pattern_xy(self):
        coefficients = []
        r_squared =[]
        best_index = 0
        x_mu, y_mu, x_sig, y_sig, x_r, y_r = self.get_gaussian(0)
        current_score = score_gaussian(x_mu, y_mu, x_sig, y_sig)
        best_score = current_score
        coefficients.append([x_mu, y_mu, x_sig, y_sig, current_score])
        r_squared.append([x_r, y_r])
        x, y = x_mu, y_mu
        for i in range(1, len(self.edm_patterns)):
            x_mu, y_mu, x_sig, y_sig, x_r, y_r = self.get_gaussian(i)
            coefficients.append([x_mu, y_mu, x_sig, y_sig])
            r_squared.append([x_r, y_r])

            current_score = score_gaussian(x_mu, y_mu, x_sig, y_sig)
            if current_score < best_score and x_r+y_r > 1.4:
                best_index = i
                best_score = current_score
                x, y = x_mu, y_mu

        return best_index, best_score, x, y

    def get_edm_cart(self):
        if self.flip:
            X_edm = self.polar_dist_flipped
        else:
            X_edm = self.polar_dist
        for i in range(len(X_edm)):
            self.edm_distribution[i,0] = X_edm[i,0]*np.cos(X_edm[i,1])
            self.edm_distribution[i,1] = X_edm[i,0]*np.sin(X_edm[i,1])

    def get_cart_error(self):
        X_edm = self.edm_distribution
        X_nn = self.nn_distribution

        Error = [np.sqrt((X_edm[i,0]-X_nn[i,0])**2+(X_edm[i,1]-X_nn[i,1])**2) for i in range(0, len(X_edm))
                 if X_edm[i][0] >0 or X_edm[i][0]<0]
        plt.figure(11, figsize=(4, 4))

        err = np.subtract(X_edm, X_nn)

        plt.plot(err[:, 0], err[:, 1], '.')

        plt.xlabel('Difference in X [cm]')
        plt.ylabel('Difference in Y [cm]')
        plt.savefig('../EDM_Support/Errs_Cart' + strftime("%m_%d_%H:%M:%S", gmtime()) + '.png')
        plt.close()

        return np.mean(Error), np.std(Error)