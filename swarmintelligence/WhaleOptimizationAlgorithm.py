from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class WhaleOptimizationAlgorithm(SwarmIntelligence):
    def __init__(self, k, n_whales, maxIteration, constanta=2, fitness_function='otsu', obj='max', initial_solution=None):
        super(WhaleOptimizationAlgorithm, self).__init__(initial_solution=initial_solution,
                                                         class_name='WhaleOptimizationAlgorithm')

        # initialize SMA parameter
        self.NUM_WHALE_ELEMENT = k
        self.WHALE_SIZE = n_whales
        self.CONSTANTA = constanta
        self.MAX_ITERATION = maxIteration
        self.OBJ = obj

        # set objective function
        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(WhaleOptimizationAlgorithm, self).fit_run(image_array)
        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        # run slime mould process
        return self.whale_optimization_algorithm()

    def whale_optimization_algorithm(self):
        # define function untuk menghitung distance vector
        def distance_vector(vector1, vector2, weight_vector1=1):
            return np.sum((weight_vector1 * vector1) - vector2)

        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        dim = self.NUM_WHALE_ELEMENT
        n_whales = self.WHALE_SIZE
        n_iterations = self.MAX_ITERATION
        fitness_function = self.FITNESS_FUNCTION
        obj = self.OBJ
        b = self.CONSTANTA

        """
        Tahap 1:
        - Inisialisasi variabel
        - Inisialisasi posisi whale
        """
        # inisialisasi best fitness dan worst fitness tracking
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        # inisialisasi posisi whale
        if self.INITIAL_SOLUTIONS != None:
            whales = np.array(self.INITIAL_SOLUTIONS)
        else:
            whales = np.random.randint(xmin, xmax,  size=(
                n_whales, dim))
        # inisialisasi fitness
        if obj == "min":
            fitness_scores = [float('inf') for _ in range(n_whales)]
        else:
            fitness_scores = [float('-inf') for _ in range(n_whales)]

        # tentukan whales teroptimal sebagai prey
        whale_best_position = whales[rd.randint(0, n_whales-1)]

        """
        Tahap 2: Whale Optimization Process
        """
        for iteration in range(n_iterations):
            # Hitung nilai a, p, dan l
            a = 2 * (1-(iteration/n_iterations))

            # update position for each whale
            for idx_whales, whale in enumerate(whales):
                p = rd.random()  # probabilitas untuk menentukan spiral model atau shrinking encircling mechanism to update position
                l = rd.uniform(-1, 1)

                # perbarui vektor A, B
                A = 2 * a * np.random.rand(dim) - a
                B = 2 * np.random.rand(dim)

                if p >= 0.5:
                    # spiral updating position (mimic the helix-shaped movement of the humpback whales around prey)
                    # distance vector
                    # dist = distance_vector(
                    #     whale_best_position, whales[idx_whales])
                    dist = np.abs(whale_best_position - whale)
                    whales[idx_whales] = dist * math.pow(math.e, (b*l)) * math.cos(
                        2*math.pi*rd.uniform(-1, 1)) + whale_best_position
                else:
                    # shrinking encircling
                    if np.sum(A) >= 1:
                        # select random search agent
                        random_whale = whales[rd.randint(0, n_whales-1)]
                        # update whale position using Eq 14 and 15
                        # dist = distance_vector(
                        #     random_whale, whales[idx_whales], weight_vector1=B)
                        dist = np.abs(B*random_whale - whale)
                        whales[idx_whales] = random_whale - (A*dist)
                    else:
                        # update whale position using Eq 8 and 9
                        # dist = distance_vector(
                        #     whale_best_position, whales[idx_whales], weight_vector1=B)
                        dist = np.abs(B*whale_best_position - whale)
                        whales[idx_whales] = whale_best_position - (A*dist)

            # set whales boundaries
            for idx_whale in range(n_whales):
                for dimension in range(dim):
                    if whales[idx_whale][dimension] < xmin:
                        whales[idx_whale][dimension] = xmin
                    elif whales[idx_whale][dimension] > xmax:
                        whales[idx_whale][dimension] = xmax - \
                            (round(rd.uniform(0, 1) * rd.randint(xmin, xmax)))

            # round and set to integer from floating number
            whales = np.round(whales).astype('int64')

            # Tahap Hitung Fitness
            fitness_scores = np.array(
                [fitness_function(whale) for whale in whales])
            if obj == 'min':
                best_index = fitness_scores.argmin()
                whale_best_position = whales[best_index][:]
                whale_best_value = fitness_scores.min()
                whale_worst_value = fitness_scores.max()
            else:
                best_index = fitness_scores.argmax()
                whale_best_position = whales[best_index][:]
                whale_best_value = fitness_scores.max()
                whale_worst_value = fitness_scores.min()
            best_fitness_tracking.append(whale_best_value)
            worst_fitness_tracking.append(whale_worst_value)

        # cari wolf alpha
        best_thresholds = np.sort(whale_best_position)

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return whales, best_thresholds
