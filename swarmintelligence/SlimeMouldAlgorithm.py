from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import math


class SlimeMouldAlgorithm(SwarmIntelligence):
    def __init__(self, k, slimeSize, maxIteration, z=0.5, fitness_function='otsu', obj='max', initial_solution=None):
        super(SlimeMouldAlgorithm, self).__init__(initial_solution=initial_solution,
                                                  class_name='SlimeMouldAlgorithm')

        # initialize SMA parameter
        self.NUM_SLIME_ELEMENT = k
        self.SLIME_SIZE = slimeSize
        self.MAX_ITERATION = maxIteration
        self.Z = z
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
        super(SlimeMouldAlgorithm, self).fit_run(image_array)
        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())

        # run slime mould process
        return self.slime_mould_algorithm()

    def slime_mould_algorithm(self):

        # inisialisasi best fitness dan worst fitness tracking
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        """
        Tahap 1: Inisialisasi
        - inisialisasi posisi dari slime moulds dalam rentang search space
        """
        if self.INITIAL_SOLUTIONS != None:
            slimeMoulds = np.array(self.INITIAL_SOLUTIONS)
        else:
            slimeMoulds = np.random.randint(self.LOWER_BOUND, self.UPPER_BOUND,  size=(
                self.SLIME_SIZE, self.NUM_SLIME_ELEMENT))

        """
        Hitung nilai Fitness
        """
        # Tahap 3: Hitung nilai fitness dari semua slime moulds
        fitness_scores = np.array(
            [self.FITNESS_FUNCTION(slime) for slime in slimeMoulds])

        # Tahap 4: Tentukan slime dengan fitness teroptimal
        if self.OBJ == 'min':
            slime_best_index = fitness_scores.argmin()
            slime_best_position = slimeMoulds[fitness_scores.argmin()][:]
            slime_best_value = fitness_scores.min()
            slime_worst_value = fitness_scores.max()
        else:
            slime_best_index = fitness_scores.argmax()
            slime_best_position = slimeMoulds[fitness_scores.argmax()][:]
            slime_best_value = fitness_scores.max()
            slime_worst_value = fitness_scores.min()

        """
        Tahap 2: Optimasi SMA
        """
        for iteration in range(self.MAX_ITERATION):
            # Tahap 5: Hitung W
            WeightVector = []
            # sort slime moulds base on their fitness values
            # if minimization then ascending order, else descending order
            slimeMouldsWithFitness = list(zip(slimeMoulds, fitness_scores))
            slimeMouldsWithFitness.sort(
                key=lambda x: x[1], reverse=self.isDescendingOrder)
            slimeMoulds = [list(slime) for slime, _ in slimeMouldsWithFitness]
            sorted_fitness_scores = [fitness for _,
                                     fitness in slimeMouldsWithFitness]
            laplace = 0
            if (slime_best_value-slime_worst_value)==0:
                laplace = np.finfo(float).eps
            for idx_sorted_fitness_scores, fitness_score in enumerate(sorted_fitness_scores):
                inside_log = (slime_best_value-fitness_score) / ((slime_best_value-slime_worst_value) + laplace)
                w = rd.uniform(0, 1) * math.log2(inside_log+1)
                if idx_sorted_fitness_scores < math.ceil(self.SLIME_SIZE/2):
                    # if S(i) ranks the first half of the population
                    w = 1 + w
                else:
                    w = 1 - w
                WeightVector.append(w)

            # Tahap 6: Updating slime mould position
            for idx_slime, slime_mould in enumerate(slimeMoulds):
                # Tahap 7: update nilai p
                p = math.tanh(abs(fitness_scores[idx_slime]-slime_best_value))

                # Tahap 8: update nilai vb dan vc
                a = math.atanh(-(iteration/self.MAX_ITERATION) +
                               np.finfo(float).eps)
                vb = np.random.uniform(
                    low=-a, high=a, size=self.NUM_SLIME_ELEMENT)
                vc = np.array([(1 - (iteration/self.MAX_ITERATION))
                              for _ in range(self.NUM_SLIME_ELEMENT)])

                # Tahap 9: Update position
                r = rd.uniform(0, 1)
                if r < p:
                    # pick 2 random slime mould from the population
                    slime_mould_random1 = np.array(
                        slimeMoulds[np.random.randint(0, self.SLIME_SIZE-1)])
                    slime_mould_random2 = np.array(
                        slimeMoulds[np.random.randint(0, self.SLIME_SIZE-1)])
                    slimeMoulds[idx_slime] = slime_mould + (
                        vb * (WeightVector[idx_slime]*slime_mould_random1 - slime_mould_random2))
                elif r >= p:
                    slimeMoulds[idx_slime] = vc * slime_mould
                else:
                    rand = rd.uniform(0, 1)
                    if rand < self.Z:
                        # Z is probability used to determine if the SMA will search for another food source (randomly)
                        slimeMoulds[idx_slime] = np.random.rand(
                            self.NUM_SLIME_ELEMENT)*(self.UPPER_BOUND-self.LOWER_BOUND)+self.LOWER_BOUND
                        # np.array(
                        #     [rand*(self.UPPER_BOUND-self.LOWER_BOUND)+self.LOWER_BOUND for _ in range(self.NUM_SLIME_ELEMENT)])

            # Check the slime moulds positions boundaries
            for idx_slime in range(self.SLIME_SIZE):
                for dimension in range(self.NUM_SLIME_ELEMENT):
                    if slimeMoulds[idx_slime][dimension] <= self.LOWER_BOUND:
                        slimeMoulds[idx_slime][dimension] = self.LOWER_BOUND
                    elif slimeMoulds[idx_slime][dimension] >= self.UPPER_BOUND:
                        slimeMoulds[idx_slime][dimension] = self.UPPER_BOUND - \
                            (round(rd.uniform(0, 1) *
                             rd.randint(self.LOWER_BOUND, self.UPPER_BOUND)))

            # round value in slimeMould position if float number
            slimeMoulds = np.round(slimeMoulds).astype('int64')

            # Tahap 3: Hitung nilai fitness dari semua slime moulds
            fitness_scores = np.array(
                [self.FITNESS_FUNCTION(slime) for slime in slimeMoulds])

            # Tahap 4: Tentukan slime dengan fitness teroptimal
            if self.OBJ == 'min':
                slime_best_index = fitness_scores.argmin()
                slime_best_position = slimeMoulds[fitness_scores.argmin()][:]
                slime_best_value = fitness_scores.min()
                slime_worst_value = fitness_scores.max()
            else:
                slime_best_index = fitness_scores.argmax()
                slime_best_position = slimeMoulds[fitness_scores.argmax()][:]
                slime_best_value = fitness_scores.max()
                slime_worst_value = fitness_scores.min()

            best_fitness_tracking.append(slime_best_value)
            worst_fitness_tracking.append(slime_worst_value)

        # set class properties variables after training phase was done
        best_thresholds = np.sort(slime_best_position)
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = slime_best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return slimeMoulds, best_thresholds
