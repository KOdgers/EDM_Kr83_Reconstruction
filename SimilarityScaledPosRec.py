import numpy as np
import numpy.linalg
from random import shuffle

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

    def get_sim_score(self, pattern, score_type):
        if not max(pattern):
            score = False
        elif score_type == 'BinaryDifference':
            sim_pattern = np.logical_xor(self.pattern, pattern)
            score = (sim_pattern == True).sum()

        elif score_type == 'NormedDifferenceSum':
            A = [i/max(self.pattern) for i in self.pattern]
            B = [j/max(pattern) for j in pattern]
            score = np.sum(np.abs(np.array(A)-np.array(B)))
        else:
            score = False

        return score

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
        self.score = 'NormedDifferenceSum'  # Other options. BinaryDifference...

    def give_events(self, events):
        if self.empty:
            print('Starting with ' +str(len(events))+ ' number of events')
            unsorted_events = [ x for i,x in enumerate(events) if max(x[0]) and i < 400]
            self.edm = np.zeros((len(unsorted_events),len(unsorted_events)))
            self.edm[0, 0] = 0
            shuffle(unsorted_events)
            unsorted_events, unsorted_x, unsorted_y = zip(*unsorted_events) # Unzip the pattern, x and y after shuffle
            unsorted_events = [x.tolist() for x in unsorted_events]
            self.pattern_list.append(Pattern(0, unsorted_events[0]))
            self.pattern_list[0].set_nn_position(unsorted_x[0], unsorted_y[0])

            temp_similarity = [self.pattern_list[0].get_sim_score(item, self.score) for item in unsorted_events]
            events_list = [x for _, x in sorted(zip(temp_similarity, unsorted_events))]
            events_x = [x for _, x in sorted(zip(temp_similarity, unsorted_x))]
            events_y = [x for _, x in sorted(zip(temp_similarity, unsorted_y))]
            print('Finished shuffling. '+str(len(unsorted_events))+ ' Events are left.')
            quarter, half, three_quarter = False, False, False
            for i,item in enumerate(events_list[1:]):
                if max(item)>0:
                    for j, item2 in enumerate(self.pattern_list[:i]):
                        self.edm[i, j] = item2.get_sim_score(item, self.score)
                        self.edm[j, i] = self.edm[i, j]
                    self.pattern_list.append(Pattern(i, item))
                    self.pattern_list[i].set_nn_position(events_x[i], events_y[i])

                if i/len(events_list)>.25 and not quarter:
                    print('Finished a quarter of EDM')
                    quarter = True
                elif i/len(events_list)>.5 and not half:
                    print('Finsiehd half of the EDM')
                    half = True
                elif i/len(events_list)>.75 and not three_quarter:
                    print('Finished Three Quarters of the EDM')
                    three_quarter = True


            self.empty = False

        else:
            #  do something with appending
            print('Please dont add events more than once at this point')

    def mds_classic(self):
        print('Starting MDS, Single Value Decomposition of the EDM')
        I = np.identity(len(self.pattern_list))
        J = I - (1/len(self.pattern_list))*np.ones(np.shape(I))
        G = np.matmul(-J, np.matmul(self.edm, J))*.5

        U, S, V = numpy.linalg.svd(G)
        print(np.shape(U), np.shape(S), np.shape(V))
        S = np.diag(S)
        X = np.matmul(np.sqrt(S), V)

        for i, item in enumerate(self.pattern_list):
            item.set_position(X[i, 0], X[i, 1])
        self.distribution = X
        print('Finished MDS')

    def get_distribution(self):
        return self.distribution

    def get_nn_distribution(self):
        X = [item.get_nn_position()[0] for item in self.pattern_list]
        Y = [item.get_nn_position()[1] for item in self.pattern_list]
        return(X, Y)




