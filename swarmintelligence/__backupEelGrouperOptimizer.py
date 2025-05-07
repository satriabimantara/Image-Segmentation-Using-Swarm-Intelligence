from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class EelGrouperOptimizer(SwarmIntelligence):
    def __init__(self, k, eelSize, maxIteration, fitness_function='otsu', obj='max', initial_solution=None):
        super(EelGrouperOptimizer, self).__init__(
            initial_solution=initial_solution, class_name='EelGrouperOptimizer')

        # initialize EGO parameter
        self.NUM_EEL_ELEMENT = k
        self.EEL_SIZE = eelSize
        self.MAX_ITERATION = maxIteration
        self.OBJ = obj
        self.isDescendingOrder = True
        if self.OBJ == 'min':
            self.isDescendingOrder = False

        # set objective function
        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(EelGrouperOptimizer, self).fit_run(image_array)

        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        return self.__optimize()

    def __optimize(self):
        """
        Function for solving ML-ISP using EGO
        """
        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        dim = self.NUM_EEL_ELEMENT
        n_eels = self.EEL_SIZE
        n_iterations = self.MAX_ITERATION
        fitness_function = self.FITNESS_FUNCTION
        isDescendingOrder = self.isDescendingOrder
        obj = self.OBJ

        # Tahap 1: inisialiasi variables
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        if self.INITIAL_SOLUTIONS != None:
            agents = np.array(self.INITIAL_SOLUTIONS)
        else:
            agents = np.random.randint(xmin, xmax,  size=(
                n_eels, dim))

        # Tahap 2: tentukan XPrey, XGrouper, XEels
        XPrey = agents[rd.randint(0, n_eels-1)].copy()
        XGrouper = agents[rd.randint(0, n_eels-1)].copy()
        XEel = agents[rd.randint(0, n_eels-1)].copy()

        # inisialisasi fitness
        if obj == "min":
            XPrey_fitness = float('inf')
            XGrouper_fitness = float('inf')
            XEel_fitness = float('inf')
            agents_fitness = np.array([float('inf') for _ in range(n_eels)])
        else:
            XPrey_fitness = float('-inf')
            XGrouper_fitness = float('-inf')
            XEel_fitness = float('-inf')
            agents_fitness = np.array([float('-inf') for _ in range(n_eels)])

        # Tahap 3: EGO Optimization process
        for iteration in range(n_iterations):
            # update a and starvation rate
            a = 2 - 2 * (iteration/n_iterations)
            starvation_rate = 100 * (iteration/n_iterations)

            # update eel position
            for idx_agent, agent in enumerate(agents):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = (a - 2) * r1 + 2
                r4 = 100 * np.random.rand()
                b = a * r2
                C1 = 2 * a * r1 - a  # Coefficient for Grouper update
                C2 = 2 * r2  # Coefficient for Eel update

                # update agent position based on grouper
                X_rand = np.array(agents[rd.randint(0, n_eels-1)].copy())
                D_grouper = abs(agent - C2*X_rand)
                agents[idx_agent] = X_rand + C1 * D_grouper

                # update Xeel position
                if r4 <= starvation_rate:
                    XEel = C2 * XGrouper
                else:
                    X_rand = np.array(agents[rd.randint(0, n_eels-1)].copy())
                    XEel = C2 * X_rand

                # update variables X1 and X2
                Distance2Eel = abs(XEel - XPrey)
                X_1 = math.exp(b*r3) * math.sin(2*math.pi*r3) * \
                    C1 * Distance2Eel + XEel

                Distance2Grouper = abs(XGrouper - XPrey)
                X_2 = XGrouper + C1 * Distance2Grouper

                if np.random.rand() < 0.5:
                    agents[idx_agent] = (0.8*X_1 + 0.2*X_2)/2
                else:
                    agents[idx_agent] = (0.2*X_1 + 0.8*X_2)/2

                # apply boundaries
                for dimension in range(dim):
                    if agents[idx_agent][dimension] < xmin:
                        agents[idx_agent][dimension] = xmin
                    elif agents[idx_agent][dimension] > xmax:
                        agents[idx_agent][dimension] = xmax - \
                            (round(rd.uniform(0, 1) * rd.randint(xmin, xmax)))
                # rounding and set to integer from floating number
                agents[idx_agent] = np.round(agents[idx_agent]).astype('int')

                # calculate fitness
                agents_fitness[idx_agent] = fitness_function(agents[idx_agent])

                # update XGrouper
                if agents_fitness[idx_agent] > XGrouper_fitness:
                    XGrouper = agents[idx_agent].copy()
                    XGrouper_fitness = agents_fitness[idx_agent]

            # keep tracking best and worst fitness
            if obj == "min":
                best_index = agents_fitness.argmin()
                best_position = agents[best_index][:]
                best_fitness_value = agents_fitness.min()
                worst_fitness_value = agents_fitness.max()
            else:
                best_index = agents_fitness.argmax()
                best_position = agents[best_index][:]
                best_fitness_value = agents_fitness.max()
                worst_fitness_value = agents_fitness.min()

            # update XPrey
            XPrey = best_position.copy()

            # append to best and worst fitness tracking
            best_fitness_tracking.append(best_fitness_value)
            worst_fitness_tracking.append(worst_fitness_value)

            # print(f"Iteration {iteration} = {best_fitness_value}")

        # cari XPrey sebagai solusi terbaik
        best_thresholds = np.sort(XPrey)

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking
        return agents, best_thresholds
