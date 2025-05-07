from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class MemoryBasedGreyWolfOptimizer(SwarmIntelligence):
    def __init__(self, k, wolfSize, maxIteration, crossover_rate=0.5, fitness_function='otsu', obj='max', initial_solution=None):
        super(MemoryBasedGreyWolfOptimizer, self).__init__(initial_solution=initial_solution,
                                                           class_name='MemoryBasedGreyWolfOptimizer')
        # initialize GWO parameter
        self.NUM_WOLF_ELEMENT = k
        self.WOLF_SIZE = wolfSize
        self.MAX_ITERATION = maxIteration
        self.OBJ = obj
        self.CR = crossover_rate
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
        super(MemoryBasedGreyWolfOptimizer, self).fit_run(image_array)
        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        # run memory based gwo
        return self.memory_grey_wolf_optimization()

    def memory_grey_wolf_optimization(self):
        xmin = self.LOWER_BOUND
        xmax = self.UPPER_BOUND
        dim = self.NUM_WOLF_ELEMENT
        n_wolfs = self.WOLF_SIZE
        n_iterations = self.MAX_ITERATION
        fitness_function = self.FITNESS_FUNCTION
        isDescendingOrder = self.isDescendingOrder
        obj = self.OBJ
        cr = self.CR
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

        # inisialisasi fitness (best values) setiap partikel PBest
        pBestpositionGreyWolfs = greyWolfs

        # inisialisasi fitness
        if obj == "min":
            pBestFitnessScores = [float('inf') for _ in range(n_wolfs)]
        else:
            pBestFitnessScores = [float('-inf') for _ in range(n_wolfs)]

        # tentukan alpha, betha, delta wolfs
        alphaWolf = greyWolfs[0]
        bethaWolf = greyWolfs[1]
        deltaWolf = greyWolfs[2]

        """
        Tahap 2: Grey Wolf Optimization Process
        """
        for iteration in range(n_iterations):
            a = 2 - 2 * iteration / n_iterations
            k = 1 - 1*iteration/n_iterations

            # Tahap Updating position for each grey wolf
            for idx_wolf, wolf in enumerate(greyWolfs):
                # get the personal best for this wolf
                pBestWolf = pBestpositionGreyWolfs[idx_wolf]

                # hitung Komponen Alpha
                A1 = 2 * a * np.random.rand(dim) - a
                C1 = 2 * np.random.rand(dim)
                D_alpha = np.abs(C1 * alphaWolf - pBestWolf)
                X1 = alphaWolf - (A1*D_alpha)

                # hitung komponen Betha
                A2 = 2 * a * np.random.rand(dim) - a
                C2 = 2 * np.random.rand(dim)
                D_betha = np.abs(C2 * bethaWolf - pBestWolf)
                X2 = bethaWolf - (A2*D_betha)

                # hitung komponen Delta
                A3 = 2 * a * np.random.rand(dim) - a
                C3 = 2 * np.random.rand(dim)
                D_deltha = np.abs(C3 * deltaWolf - pBestWolf)
                X3 = deltaWolf - (A3*D_deltha)

                # update posisi Wolf ke-i berdasarkan crossover
                if rd.uniform(0, 1) < cr:
                    # update posisi Wolf ke-i based on new hunting mechanism
                    greyWolfs[idx_wolf] = (X1+X2+X3)/3
                else:
                    # NOVEL: explore and retrace the neighbourhood areas of the personal best states of wolves
                    random_wolf_1 = np.array(
                        greyWolfs[rd.randint(0, n_wolfs-1)])
                    random_wolf_2 = np.array(
                        greyWolfs[rd.randint(0, n_wolfs-1)])
                    greyWolfs[idx_wolf] = pBestWolf + k * \
                        np.abs(random_wolf_1 - random_wolf_2)

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

            # Update personal best memory of grey wolves
            for idx_wolf, wolf in enumerate(greyWolfs):
                if ((fitness_scores[idx_wolf] > pBestFitnessScores[idx_wolf]) and (obj == 'max')) or ((fitness_scores[idx_wolf] < pBestFitnessScores[idx_wolf]) and (obj == 'min')):
                    pBestpositionGreyWolfs[idx_wolf] = greyWolfs[idx_wolf]
                    pBestFitnessScores[idx_wolf] = fitness_scores[idx_wolf]

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
            print(f"Iteration {iteration} = {wolf_best_value}")

        # cari wolf alpha sebagai solusi terbaik
        best_thresholds = np.sort(alphaWolf)

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return greyWolfs, best_thresholds
