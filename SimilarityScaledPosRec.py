import numpy as np
import numpy.linalg
import numexpr as ne
from random import shuffle, randint
import matplotlib.pyplot as plt

from time import gmtime, strftime

from matplotlib.collections import LineCollection

from sklearn import manifold

from TPC_Configuration import active_pmts
# from sklearn.decomposition import PCA

class Pattern(object):

    def __init__(self, id, pattern):
        self.pattern = pattern
        self.x_pos = False
        self.y_pos = False
        self.x_nn = False
        self.y_nn = False
        self.id = id

    def get_pattern(self):
        return self.pattern

    def get_sim_score(self, pattern, score_type,pmt_selection):

        ## Implement Quantum Efficiencies if not MonteCarlo Derived

        if not max(pattern):
            score = False
        elif score_type == 'BinaryDifference':
            sim_pattern = np.logical_xor(self.pattern, pattern)
            score = (sim_pattern == True).sum()

        elif score_type == 'Manhattan_Distance':
            A = [i / max(self.pattern) for i in self.pattern]  ## Should I use Sum or Max for Normalizaiton
            B = [j / max(pattern) for j in pattern]
            score = np.sum(np.abs(np.array(A) - np.array(B)))

        elif score_type == 'RMS_Normed':
            A = [i / max(self.pattern) for i in self.pattern]  ## Should I use Sum or Max for Normalizaiton
            B = [j / max(pattern) for j in pattern]
            score = np.sqrt(np.sum(np.square(np.subtract(np.array(A), np.array(B)))))
        elif score_type == 'Pattern_Fitter':
            score = self.get_gof_pattern_fitter(pattern,pmt_selection, 'likelihood_poisson')

        #### I need to implement Profile Likelihood Function from XENON Code

        else:
            score = False

        return score

    def get_gof_pattern_fitter(self, pattern, pmt_selection, statistic):

        areas_observed = np.array([item for i, item in enumerate(pattern) if i in pmt_selection])
        q = np.array([item for i, item in enumerate(self.pattern) if i in pmt_selection])
        square_syst_errors = (0 * areas_observed) ** 2

        qsum = q.sum(axis=-1)[..., np.newaxis]  # noqa
        fractions_expected = ne.evaluate("q / qsum")  # noqa
        total_observed = areas_observed.sum()  # noqa
        ao = areas_observed  # noqa
        # square_syst_errors = square_syst_errors[pmt_selection]  # noqa

        # The actual goodness of fit computation is here...
        # Areas expected = fractions_expected * sum(areas_observed)
        if statistic == 'chi2gamma':
            result = ne.evaluate("(ao + where(ao > 1, 1, ao) - {ae})**2 /"
                                 "({ae} + square_syst_errors + 1)".format(ae='fractions_expected * total_observed'))
        elif statistic == 'chi2':
            result = ne.evaluate("(ao - {ae})**2 /"
                                 "({ae} + square_syst_errors)".format(ae='fractions_expected * total_observed'))
        elif statistic == 'likelihood_poisson':
            # Poisson likelihood chi-square (Baker and Cousins, 1984)
            # Clip areas to range [0.0001, +inf), because of log(0)
            areas_expected_clip = np.clip(fractions_expected * total_observed, 1e-10, float('inf'))
            areas_observed_clip = np.clip(areas_observed, 1e-10, float('inf'))
            result = ne.evaluate("-2*({ao} * log({ae}/{ao}) + {ao} - {ae})".format(ae='areas_expected_clip',
                                                                                   ao='areas_observed_clip'))
        else:
            raise ValueError('Pattern goodness of fit statistic %s not implemented!' % statistic)

        return np.sum(result, axis=-1)



    def set_position(self, x, y):
        self.x_pos = x
        self.y_pos = y

    def set_nn_position(self, x, y):
        self.x_nn = x
        self.y_nn = y

    def get_position(self):
        return self.x_pos, self.y_pos

    def get_nn_position(self):
        return self.x_nn, self.y_nn


class TpcRep(object):

    def __init__(self):
        self.edm = np.zeros(1)
        self.resolution = 5
        self.pattern_list = []
        self.distribution = np.array(1)
        self.empty = True
        self.score = 'Pattern_Fitter'  # 'RMS_Normed' 'Manhattan_Distance'  # Other options. BinaryDifference...
        self.tpc_radius = 50
        self.polar_dist = np.array(1)
        self.polar_dist_flipped = np.array(1)
        self.polar_nn_dist = np.array(1)
        self.pmt_selection = active_pmts

    def give_events(self, events):
        if self.empty:
            print('Starting with ' + str(len(events)) + ' number of events')
            number_of_events = 400
            event_start = randint(100, len(events)-number_of_events)
            unsorted_events = [x for i, x in enumerate(events) if max(x[0]) and event_start < i < (event_start+number_of_events)]
            self.edm = np.zeros((len(unsorted_events), len(unsorted_events)))
            self.edm[0, 0] = 0
            shuffle(unsorted_events)
            unsorted_events, unsorted_x, unsorted_y = zip(*unsorted_events)  # Unzip the pattern, x and y after shuffle
            unsorted_events = [x.tolist()[0:127] for x in unsorted_events]
            self.pattern_list.append(Pattern(0, unsorted_events[0]))
            self.pattern_list[0].set_nn_position(unsorted_x[0], unsorted_y[0])

            temp_similarity = [self.pattern_list[0].get_sim_score(item, self.score, active_pmts) for item in unsorted_events]
            events_list = [x for _, x in sorted(zip(temp_similarity, unsorted_events))]
            events_x = [x for _, x in sorted(zip(temp_similarity, unsorted_x))]
            events_y = [x for _, x in sorted(zip(temp_similarity, unsorted_y))]
            print('Finished shuffling. ' + str(len(unsorted_events)) + ' Events are left.')
            quarter, half, three_quarter = False, False, False
            for i, item in enumerate(events_list[1:]):
                if max(item) > 0:
                    for j, item2 in enumerate(self.pattern_list[:i]):
                        self.edm[i, j] = item2.get_sim_score(item, self.score, active_pmts)
                        self.edm[j, i] = self.edm[i, j]
                    self.pattern_list.append(Pattern(i, item))
                    self.pattern_list[i].set_nn_position(events_x[i], events_y[i])

                if i / len(events_list) > .25 and not quarter:
                    print('Finished a quarter of EDM')
                    quarter = True
                elif i / len(events_list) > .5 and not half:
                    print('Finsiehd half of the EDM')
                    half = True
                elif i / len(events_list) > .75 and not three_quarter:
                    print('Finished Three Quarters of the EDM')
                    three_quarter = True

            self.empty = False

        else:
            #  do something with appending
            print('Please dont add events more than once at this point')

    def cut_worse_5_percent(self):
        mean_score = np.mean(self.edm)
        mean_score_array = np.mean(self.edm, axis=1)
        dev_score = np.std(mean_score_array)
        del_count = 0
        for i, score in enumerate(mean_score_array.tolist()):
            if score > mean_score + dev_score:
                del self.pattern_list[i - del_count]
                self.edm = np.delete(self.edm, i - del_count, 0)
                self.edm = np.delete(self.edm, i - del_count, 1)
                del_count += 1

        print('There are ' + str(len(self.pattern_list)) + ' events left after cuts')

    def mds_classic(self):
        print('Starting MDS, Single Value Decomposition of the EDM')
        I = np.identity(len(self.pattern_list))
        J = I - np.ones(np.shape(I))
        G = np.matmul(-J, np.matmul(self.edm, J)) * .5
        U, S, V = numpy.linalg.svd(G)
        print(np.shape(U), np.shape(S), np.shape(V))
        S = np.diag(S)
        X = np.matmul(np.sqrt(S), V)
        for i, item in enumerate(self.pattern_list):
            item.set_position(X[i, 0], X[i, 1])
        self.distribution = X[:, 0:2]
        print('Finished MDS')

    def sklearn_mds(self):
        seed = np.random.RandomState(seed=3)
        nmds = manifold.MDS(n_components=2, metric=False, max_iter=1000, eps=1e-12,
                            dissimilarity="precomputed", random_state=seed, n_jobs=1,
                            n_init=1)

        mds = manifold.MDS(n_components=2, max_iter=1000, eps=1e-9, random_state=seed,
                           dissimilarity="precomputed", n_jobs=1,n_init = 10)
        pos = mds.fit(self.edm).embedding_

        npos = nmds.fit_transform(self.edm, init=pos)
        plt.figure(10)
        plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', lw=0, label='NMDS')
        plt.title('Quick n Dirty Sklearn')
        plt.savefig('../EDM_Support/Sklearn' + strftime("%m_%d_%H:%M:%S", gmtime()) + '.png')

        self.distribution = npos

    def get_distribution(self):
        return self.distribution

    def get_nn_distribution(self):
        X = [item.get_nn_position()[0] for item in self.pattern_list]
        Y = [item.get_nn_position()[1] for item in self.pattern_list]

        return np.column_stack((X, Y))

    def get_nn_lists(self):
        X = [item.get_nn_position()[0] for item in self.pattern_list]
        Y = [item.get_nn_position()[1] for item in self.pattern_list]

        return X, Y

    def get_patterns(self):
        return self.pattern_list

    def plot_edm_recon(self, cuts=True):
        data = self.get_distribution()

        data_split = [[data[i, 0], data[i, 1]] for i in range(0, len(data))]
        X = [item[0] for item in data_split]
        Y = [item[1] for item in data_split]
        plt.figure(1, figsize=(4, 4))
        plt.plot(X, Y, '.')
        plt.xlabel('X - unscaled [N/A]')
        plt.ylabel('Y - unscaled [N/A]')
        plt.title('Unscaled EDM Reconstructed Distribution')
        time = strftime("%m_%d_%H:%M:%S", gmtime())
        plt.savefig('../EDM_Support/FuckIt' + time + '.png')

        if cuts:
            for l in range(0, 2):
                x_mean = np.mean(X)
                y_mean = np.mean(Y)
                x_sig = np.std(X)
                y_sig = np.std(Y)
                # print(len(data),len(data[0]))
                data_cut = [[X[i], Y[i]] for i in range(0, len(X)) if
                            (np.abs(X[i]) < x_mean + 2 * x_sig) and
                            (np.abs(Y[i]) < y_mean + 2 * y_sig)]
                print(len(data_cut), len(data_cut[0]))
                X = [item[0] for item in data_cut]
                Y = [item[1] for item in data_cut]

                x_mean = np.mean(X)
                y_mean = np.mean(Y)
                x_scale = 100 / (max(X) - min(X))
                y_scale = 100 / (max(Y) - min(Y))

                # x_scale, y_scale = 1, 1

                X = ([(item - x_mean) * x_scale for item in X])
                Y = ([(item - y_mean) * y_scale for item in Y])

        plt.figure(2, figsize=(4, 4))
        plt.plot(X, Y, '.')
        plt.xlabel('X - unscaled [N/A]')
        plt.ylabel('Y - unscaled [N/A]')
        plt.title('Scaled EDM Reconstructed Distribution')
        time = strftime("%m_%d_%H:%M:%S", gmtime())
        plt.savefig('../EDM_Support/FuckItLetsGo' + time + '.png')

    def plot_edm_nn(self, ptype):
        # Rescale the data
        X_nn = self.get_nn_distribution()
        if ptype == 'cart':
            npos = self.distribution
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
                            zorder=0, cmap=plt.cm.Blues)#,
                            #norm=plt.Normalize(0, values.max()))
        # lc.set_array(similarities.flatten())
        lc.set_linewidths(2 * np.ones(len(segs)))
        ax.add_collection(lc)
        plt.title(the_title)
        plt.savefig('../EDM_Support/' + ptype + strftime("%m_%d_%H:%M:%S", gmtime()) + '.png')
        plt.close()

    def translation_scaling_rotation(self):
        X_nn = self.get_nn_distribution()
        npos = self.distribution
        self.polar_dist = np.zeros_like(self.distribution)
        self.polar_dist_flipped = np.zeros_like(self.distribution)
        self.polar_nn_dist = np.zeros_like(self.distribution)

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
                    theta = 2*np.pi-theta
            elif npos[i,0]<=0:
                if npos[i, 1] > 0:
                    theta = np.pi - theta
                else:
                    theta = np.pi + theta

            if X_nn[i, 0] > 0:
                if X_nn[i, 1] < 0:
                    theta2 = 2*np.pi-theta2

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
            self.polar_dist[i, 1] = theta%(2*np.pi)
            self.polar_nn_dist[i, 1] = theta2%(2*np.pi)
            self.polar_dist_flipped[i, 1] = 2*np.pi-1 * self.polar_dist[i, 1]
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

    def get_polar_errors(self):
        plt.figure(11, figsize=(9, 4))
        plt.title('Errors (Un-mirrored, Mirrored)')
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
                ang_err = np.pi-np.abs(np.abs(edm_dist[i, 1]-self.polar_nn_dist[i, 1])-np.pi)
                direction = 1
                if (np.abs(edm_dist[i,1]-self.polar_nn_dist[i,1])>(edm_dist[i,1]-self.polar_nn_dist[i,1]) and
                    np.abs(edm_dist[i, 1] - self.polar_nn_dist[i, 1])>ang_err):
                    direction = -1

                theta_err[i] = direction*ang_err
                rad_err[i] = edm_dist[i, 0]-self.polar_nn_dist[i,0]

            plt.subplot(1, 2, j+1)
            plt.xlabel('Difference in Radii [cm]')
            plt.ylabel('Error in Angle [rad]')
            plt.plot(rad_err, theta_err, '.')

        plt.xlabel('Difference in Radii [cm]')
        plt.ylabel('Error in Angle [rad]')
        plt.savefig('../EDM_Support/Errs'+strftime("%m_%d_%H:%M:%S", gmtime())+'.png')
        plt.close()
        return



