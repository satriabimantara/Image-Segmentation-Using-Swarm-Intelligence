from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class GreaterCaneRatAlgorithm(SwarmIntelligence):
    def __init__(self, k, ratSize, maxIteration, rho=0.5, fitness_function='otsu', obj='max', initial_solution=None):
        super(GreaterCaneRatAlgorithm, self).__init__(initial_solution=initial_solution,
                                                      class_name='GreaterCaneRatAlgorithm')

        # initialize GWO parameter
        self.NUM_RAT_ELEMENT = k
        self.RAT_SIZE = ratSize
        self.MAX_ITERATION = maxIteration
        self.OBJ = obj
        self.isDescendingOrder = True
        if self.OBJ == 'min':
            self.isDescendingOrder = False
        # initialize GWO improved parameter
        self.rho = rho

        # set objective function
        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(GreaterCaneRatAlgorithm, self).fit_run(image_array)
        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        # run slime mould process
        return self.greater_cane_rat_algorithm()

    def greater_cane_rat_algorithm(self):
        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        dim = self.NUM_RAT_ELEMENT
        n_rats = self.RAT_SIZE
        n_iterations = self.MAX_ITERATION
        fitness_function = self.FITNESS_FUNCTION
        isDescendingOrder = self.isDescendingOrder
        obj = self.OBJ
        rho = self.rho

        """
        Tahap 1:
        - Inisialisasi variabel
        - Inisialisasi posisi 
        """
        # inisialisasi best fitness dan worst fitness tracking
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        # Tahap 1: inisialisasi posisi
        if self.INITIAL_SOLUTIONS != None:
            agents = np.array(self.INITIAL_SOLUTIONS)
        else:
            agents = np.random.randint(xmin, xmax,  size=(
                n_rats, dim))

        # Tahap 2: calculate fitness and select dominant male
        fitness_scores = np.array(
            [fitness_function(agent) for agent in agents])
        best_index_agent = np.argmax(fitness_scores)
        if obj == 'min':
            best_index_agent = np.argmin(fitness_scores)
        dominant_male = {
            'position': agents[best_index_agent].copy(),
            'fitness': fitness_scores[best_index_agent]
        }

        # Tahap 3: Update the remaining GR based on Xk using Eq 3
        # agents = self.__update_gcr_position_eq3(agents, dominant_male)

        """
        Tahap 3: GCRA optimization process
        """
        for iteration in range(n_iterations):
            # update parameters
            params = self.__update_parameters(dominant_male, iteration)
            C = params['C']
            r = params['r']
            alpha = params['alpha']
            miu = params['miu']
            betha = params['betha']

            # update gcr position (exploitation vs exploration)
            for idx_agent, agent in enumerate(agents):
                for dimension in range(dim):
                    if np.random.rand() < rho:
                        # exploration
                        agents[idx_agent][dimension] = 0.7 * \
                            ((agents[idx_agent][dimension] +
                             dominant_male['position'][dimension])/2)
                        agents[idx_agent][dimension] = agents[idx_agent][dimension] + C * \
                            (dominant_male['position'][dimension] -
                             r * agents[idx_agent][dimension])
                    else:
                        # exploitation
                        female_rat = agents[np.random.randint(
                            0, n_rats)].copy()
                        agents[idx_agent][dimension] = agents[idx_agent][dimension] + C * \
                            (dominant_male['position'][dimension] -
                             miu * female_rat[dimension])

                # ensure the position are within the bound
                agents[idx_agent] = self.__apply_boundaries(agents[idx_agent])

                # evaluate fitness and update the dominant male
                fitness_scores[idx_agent] = fitness_function(agents[idx_agent])
                if fitness_scores[idx_agent] > dominant_male['fitness']:
                    dominant_male = {
                        'position': agents[idx_agent].copy(),
                        'fitness': fitness_scores[idx_agent]
                    }

            if obj == 'min':
                best_index = fitness_scores.argmin()
                best_position = agents[best_index][:]
                best_fitness_value = fitness_scores.min()
                worst_fitness_value = fitness_scores.max()
            else:
                best_index = fitness_scores.argmax()
                best_position = agents[best_index][:]
                best_fitness_value = fitness_scores.max()
                worst_fitness_value = fitness_scores.min()

            # append to best and worst fitness tracking
            best_fitness_tracking.append(best_fitness_value)
            worst_fitness_tracking.append(worst_fitness_value)

        # cari wolf alpha
        best_thresholds = np.sort(dominant_male['position'])

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return agents, best_thresholds

    def __update_parameters(self, dominant_male, iteration):
        MAX_ITERATIONS = self.MAX_ITERATION
        r = dominant_male['fitness'] - iteration * \
            (dominant_male['fitness']/MAX_ITERATIONS)
        miu = np.random.randint(1, 5)  # constant between 1 to 4 (inclusive)
        C = np.random.rand()
        alpha = 2 * r * np.random.rand() - r
        betha = 2 * r * miu - r
        params = {
            'C': C,
            'r': r,
            'miu': miu,
            'alpha': alpha,
            'betha': betha,
        }
        return params

    def __update_gcr_position_eq3(self, agents, dominant_male):
        for idx_agent, agent in enumerate(agents):
            agents[idx_agent] = 0.7 * ((agent+dominant_male['position'])/2)
        return agents

    def __apply_boundaries(self, agent):
        dim = self.NUM_RAT_ELEMENT
        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        temp_agent = agent.copy()
        for dimension in range(dim):
            if temp_agent[dimension] < xmin:
                temp_agent[dimension] = xmin
            elif temp_agent[dimension] > xmax:
                temp_agent[dimension] = xmax - \
                    (round(rd.uniform(0, 1) * rd.randint(xmin, xmax)))
        temp_agent = np.round(temp_agent).astype('int64')
        return temp_agent
