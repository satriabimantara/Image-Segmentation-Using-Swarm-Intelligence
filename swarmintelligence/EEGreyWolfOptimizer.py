from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class EEGreyWolfOptimizer(SwarmIntelligence):
    def __init__(self, k, wolfSize, maxIteration, a_initial=2, a_final=0, modulation_index=1.5, b1=0.1, b2=0.9, fitness_function='otsu', obj='max', initial_solution=None):
        super(EEGreyWolfOptimizer, self).__init__(initial_solution=initial_solution,
                                                  class_name='EEGreyWolfOptimizer')

        # initialize GWO parameter
        self.NUM_WOLF_ELEMENT = k
        self.WOLF_SIZE = wolfSize
        self.MAX_ITERATION = maxIteration
        self.OBJ = obj
        self.isDescendingOrder = True
        if self.OBJ == 'min':
            self.isDescendingOrder = False
        # initialize GWO improved parameter
        self.A_initial = a_initial
        self.A_final = a_final
        self.B1 = b1
        self.B2 = b2
        self.ModulationIndex = modulation_index

        # set objective function
        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(EEGreyWolfOptimizer, self).fit_run(image_array)
        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        # run slime mould process
        return self.exploration_enhanced_grey_wolf_optimization()

    def exploration_enhanced_grey_wolf_optimization(self):
        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        dim = self.NUM_WOLF_ELEMENT
        n_wolfs = self.WOLF_SIZE
        n_iterations = self.MAX_ITERATION
        fitness_function = self.FITNESS_FUNCTION
        isDescendingOrder = self.isDescendingOrder
        obj = self.OBJ
        a_initial = self.A_initial
        a_final = self.A_final
        b1 = self.B1
        b2 = self.B2
        modulation_index = self.ModulationIndex

        """
        Tahap 1:
        - Inisialisasi variabel
        - Inisialisasi posisi grey wolf
        """
        # inisialisasi best fitness dan worst fitness tracking
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        # inisialisasi posisi grey wolf
        if self.INITIAL_SOLUTIONS != None:
            greyWolfs = np.array(self.INITIAL_SOLUTIONS)
        else:
            greyWolfs = np.random.randint(xmin, xmax,  size=(
                n_wolfs, dim))

        # inisialisasi fitness
        if obj == "min":
            fitness_scores = [float('inf') for _ in range(n_wolfs)]
        else:
            fitness_scores = [float('-inf') for _ in range(n_wolfs)]

        # tentukan alpha, betha, delta wolfs
        alphaWolf = greyWolfs[0]
        bethaWolf = greyWolfs[1]
        deltaWolf = greyWolfs[2]

        """
        Tahap 2: Grey Wolf Optimization Process
        """
        for iteration in range(n_iterations):
            # new formulation for updating a
            modulation = (n_iterations-iteration)/n_iterations
            a = a_initial - (a_initial - a_final) * \
                math.pow(modulation, modulation_index)

            # Tahap Updating position grey wolf
            for idx_wolf, wolf in enumerate(greyWolfs):
                # hitung Komponen Alpha
                A1 = 2 * a * np.random.rand(dim) - a
                C1 = 2 * np.random.rand(dim)
                D_alpha = np.abs(C1 * alphaWolf - wolf)
                X1 = alphaWolf - (A1*D_alpha)

                # hitung komponen Betha
                A2 = 2 * a * np.random.rand(dim) - a
                C2 = 2 * np.random.rand(dim)
                D_betha = np.abs(C2 * bethaWolf - wolf)
                X2 = bethaWolf - (A2*D_betha)

                # hitung komponen Delta
                A3 = 2 * a * np.random.rand(dim) - a
                C3 = 2 * np.random.rand(dim)
                D_deltha = np.abs(C3 * deltaWolf - wolf)
                X3 = deltaWolf - (A3*D_deltha)

                # update each greywolf position
                random_selected_wolf = np.array(
                    greyWolfs[rd.randint(0, n_wolfs-1)])
                greyWolfs[idx_wolf] = (b1 * rd.uniform(0, 1) * (X1+X2+X3)/3) + (
                    b2 * rd.uniform(0, 1) * np.abs(random_selected_wolf - greyWolfs[idx_wolf]))

            # set grey wolf position boundaries
            for idx_wolf in range(n_wolfs):
                for dimension in range(dim):
                    if greyWolfs[idx_wolf][dimension] < xmin:
                        greyWolfs[idx_wolf][dimension] = xmin
                    elif greyWolfs[idx_wolf][dimension] > xmax:
                        greyWolfs[idx_wolf][dimension] = xmax - \
                            (round(rd.uniform(0, 1) * rd.randint(xmin, xmax)))

            # round and set to integer from floating number
            greyWolfs = np.round(greyWolfs).astype('int64')

            # Tahap Hitung Fitness
            fitness_scores = np.array(
                [fitness_function(wolf) for wolf in greyWolfs])

            if obj == 'min':
                best_index = fitness_scores.argmin()
                wolf_best_position = greyWolfs[best_index][:]
                wolf_best_value = fitness_scores.min()
                wolf_worst_value = fitness_scores.max()
            else:
                best_index = fitness_scores.argmax()
                wolf_best_position = greyWolfs[best_index][:]
                wolf_best_value = fitness_scores.max()
                wolf_worst_value = fitness_scores.min()

            # Tahap 3 Identifikasi Alpha, Betha, Delta Wolf
            # Sort greyWolfs berdasarkan nilai fitness teroptimal
            greyWolfsWithFitness = list(zip(greyWolfs, fitness_scores))
            greyWolfsWithFitness.sort(
                key=lambda x: x[1], reverse=isDescendingOrder)
            greyWolfs = [list(wolf) for wolf, _ in greyWolfsWithFitness]
            sorted_fitness_scores = [fitness for _,
                                     fitness in greyWolfsWithFitness]

            # Tentukan alpha, betha, delta
            alphaWolf = greyWolfs[0]
            bethaWolf = greyWolfs[1]
            deltaWolf = greyWolfs[2]

            # append to best and worst fitness tracking
            best_fitness_tracking.append(wolf_best_value)
            worst_fitness_tracking.append(wolf_worst_value)

        # cari wolf alpha
        best_thresholds = np.sort(alphaWolf)

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return greyWolfs, best_thresholds
