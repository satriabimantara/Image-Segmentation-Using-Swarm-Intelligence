from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class ArchimedesOptimizationAlgorithm(SwarmIntelligence):
    def __init__(self, k, objectSize, maxIteration, c1=2, c2=6, c3=2, c4=0.5, exploration_rate=0.5, fitness_function='otsu', obj='max', initial_solution=None):
        super(ArchimedesOptimizationAlgorithm, self).__init__(
            initial_solution=initial_solution, class_name='Archimedes Optimization Algorithm')

        # initialize AOA parameters
        self.NUM_OBJECT_ELEMENT = k
        self.OBJECT_SIZE = objectSize
        self.MAX_ITERATION = maxIteration
        self.OBJ = obj
        self.isDescendingOrder = True
        if self.OBJ == 'min':
            self.isDescendingOrder = False
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.exploration_rate = exploration_rate

        # set objective function
        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(ArchimedesOptimizationAlgorithm, self).fit_run(image_array)

        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        # run slime mould process
        return self.__optimize()

    def __optimize(self):
        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        dim = self.NUM_OBJECT_ELEMENT
        n_objects = self.OBJECT_SIZE
        n_iterations = self.MAX_ITERATION
        fitness_function = self.FITNESS_FUNCTION
        obj = self.OBJ

        # Tahap 1: inisialiasi variables
        # inisialisasi best fitness dan worst fitness tracking
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        # inisialiasi posisi objects
        if self.INITIAL_SOLUTIONS != None:
            agents = np.array(self.INITIAL_SOLUTIONS)
        else:
            agents = [None for _ in range(n_objects)]
            for idx_agent in range(n_objects):
                agents[idx_agent] = {
                    'position': np.random.randint(xmin, xmax,  size=(1, dim))[0],
                    'densities': np.array([np.random.random() for _ in range(dim)]),
                    'volume': np.array([np.random.random() for _ in range(dim)]),
                    'acceleration': np.array([
                        np.random.random() * (xmax - xmin) + xmin
                        for d in range(dim)
                    ])
                }

        # Tahap2: evaluate fitness and select the best object
        fitness_scores = np.array(
            [fitness_function(agent['position']) for agent in agents])
        best_index_agent = np.argmax(fitness_scores)
        if obj == 'min':
            best_index_agent = np.argmin(fitness_scores)
        best_object = agents[best_index_agent].copy()

        # Tahap 3: AOA process
        for iteration in range(n_iterations):
            # Tahap 4: update object position
            agents = self.__update_objects_position(
                agents, best_object, iteration)

            # Tahap 5: apply boundaries
            agents = self.__apply_boundaries(agents)

            # Tahap 6: evaluate object fitness
            fitness_scores = np.array(
                [fitness_function(agent['position']) for agent in agents])
            if obj == 'min':
                best_index = fitness_scores.argmin()
                best_position = agents[best_index]['position'][:]
                best_fitness_value = fitness_scores.min()
                worst_fitness_value = fitness_scores.max()
            else:
                best_index = fitness_scores.argmax()
                best_position = agents[best_index]['position'][:]
                best_fitness_value = fitness_scores.max()
                worst_fitness_value = fitness_scores.min()

            # Tahap 7: update best object with best fitness
            best_object = agents[best_index].copy()

            # append to best and worst fitness tracking
            best_fitness_tracking.append(best_fitness_value)
            worst_fitness_tracking.append(worst_fitness_value)

        # cari best object
        best_thresholds = np.sort(best_object['position'])

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return agents, best_thresholds

    def __update_objects_position(self, agents, best_object, iteration):
        def normalize_acceleration(vector_acceleration):
            U = 0.9
            L = 0.1
            normalize_vector = U * ((vector_acceleration - np.min(vector_acceleration))/(
                np.max(vector_acceleration) - np.min(vector_acceleration))) + L
            return normalize_vector
        N_OBJECT = self.OBJECT_SIZE
        MAX_ITERATIONS = self.MAX_ITERATION
        EXPLORATION_RATE = self.exploration_rate
        C1 = self.c1
        C2 = self.c2
        C3 = self.c3
        C4 = self.c4

        for idx_agent, agent in enumerate(agents):
            # update density and volume of each object
            agents[idx_agent]['densities'] = agent['densities'] + np.random.random() * \
                (best_object['densities'] - agent['densities'])
            agents[idx_agent]['volume'] = agent['volume'] + \
                np.random.random() * (best_object['volume'] - agent['volume'])

            # update transfer and density decreasing factors TF and d
            TF = math.exp((iteration - MAX_ITERATIONS)/MAX_ITERATIONS)
            d = math.exp((MAX_ITERATIONS-iteration) /
                         MAX_ITERATIONS) - (iteration/MAX_ITERATIONS)

            # check exploration or exploitation based on TF
            if TF <= EXPLORATION_RATE:
                # exploration
                # get random agent
                random_agent = agents[rd.randint(0, N_OBJECT-1)].copy()
                # update acceleration
                agents[idx_agent]['acceleration'] = (random_agent['densities'] + random_agent['volume'] *
                                                     random_agent['acceleration'])/(agents[idx_agent]['densities'] * agents[idx_agent]['volume'])
                # normalize acceleration
                agents[idx_agent]['acceleration'] = normalize_acceleration(
                    agents[idx_agent]['acceleration'])

                # update object position
                agents[idx_agent]['position'] = agent['position'] + C1 * np.random.random(
                ) * agent['acceleration'] * d * (random_agent['position'] - agent['position'])

            else:
                # exploitation
                # update acceleration
                agents[idx_agent]['acceleration'] = (best_object['densities'] + best_object['volume'] * best_object['acceleration'])/(
                    agents[idx_agent]['densities'] * agents[idx_agent]['volume'])

                # normalize acceleration
                agents[idx_agent]['acceleration'] = normalize_acceleration(
                    agents[idx_agent]['acceleration'])

                # update direction flag F
                P = 2 * np.random.random() - C4
                F = -1
                if P <= 0.5:
                    F = 1

                # update position
                T = C3 * TF
                agents[idx_agent]['position'] = best_object['position'] + F * C2 * np.random.random(
                ) * agent['acceleration'] * d * (T * best_object['position'] - agent['position'])

        return agents

    def __apply_boundaries(self, agents):
        dim = self.NUM_OBJECT_ELEMENT
        xmax = self.UPPER_BOUND
        xmin = self.LOWER_BOUND
        for idx_agent, agent in enumerate(agents):
            for dimension in range(dim):
                if agents[idx_agent]['position'][dimension] < xmin:
                    agents[idx_agent]['position'][dimension] = xmin
                elif agents[idx_agent]['position'][dimension] > xmax:
                    agents[idx_agent]['position'][dimension] = xmax - \
                        (round(rd.uniform(0, 1) * rd.randint(xmin, xmax)))
            agents[idx_agent]['position'] = np.round(
                agents[idx_agent]['position']).astype('int64')
        return agents
