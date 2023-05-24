from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np
import collections


class GeneticAlgorithm(SwarmIntelligence):
    def __init__(self, k, pop_size, max_gens, cr, mr,  fitness_function='otsu', initial_solution=None):
        super(GeneticAlgorithm, self).__init__(
            class_name='GeneticAlgorithm', initial_solution=initial_solution)
        # initialiaze Object parameter
        self.NUM_GENES = k
        self.POP_SIZE = pop_size
        self.MAX_GENERATION = max_gens
        self.MUTATION_RATE = mr
        self.CROSSOVER_RATE = cr

        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(GeneticAlgorithm, self).fit_run(image_array)
        self.MIN_VAL = min(image_array.ravel())
        self.MAX_VAL = max(image_array.ravel())
        # run genetic algorithm process
        return self.genetic_algorithm()

    def genetic_algorithm(self):
        # define genetic operator (crossover and mutation)
        def mutation(populations):
            n_mutation = round(self.POP_SIZE * self.MUTATION_RATE)
            for _ in range(n_mutation):
                # tentukan kromosom yang akan dimutasi
                idx_mutated_kromosom = rd.randint(0, self.POP_SIZE-1)
                # tentukan gen yang terkena mutasi dari kromosom yang akan dimutasi
                idx_gen_mutated = rd.randint(0, self.NUM_GENES-1)
                # mutasi gen pada kromosom
                populations[idx_mutated_kromosom][idx_gen_mutated] = int(rd.randint(
                    self.MIN_VAL, self.MAX_VAL-1))
            return populations

        def crossover(populations):
            def check_validity_child_chromosome(child_chromosome):
                count_number_element = collections.Counter(child_chromosome)
                jum_elemen = np.sum(list(count_number_element.values()))
                while jum_elemen > len(count_number_element):
                    for idx_a, element_a in enumerate(child_chromosome):
                        if count_number_element[element_a] > 1:
                            bil_random = rd.randint(
                                self.MIN_VAL, self.MAX_VAL-1)
                            while bil_random == element_a:
                                bil_random = rd.randint(
                                    self.MIN_VAL, self.MAX_VAL-1)
                            child_chromosome[idx_a] = bil_random
                            count_number_element[bil_random] += 1
                            count_number_element[element_a] -= 1
                    jum_elemen = np.sum(list(count_number_element.values()))
                return child_chromosome

            # Tahap 1: create matting pool
            indices_matting_pool = []
            matting_pool = []
            # Bangkitkan bilangan random sebanyak ukuran populasi
            random_number_of_population = np.random.random(self.POP_SIZE)
            # pilih kromosom ke-i jika random(i) < cr
            for i in range(self.POP_SIZE):
                if random_number_of_population[i] < self.CROSSOVER_RATE:
                    matting_pool.append(populations[i])
                    indices_matting_pool.append(i)

            # kalau jumlah parent pada matting pool ganjil, maka pop agar bisa kawin silang
            if len(matting_pool) % 2 == 1:
                matting_pool.pop()
                indices_matting_pool.pop()

            # Tahap 2: single point crossover
            # get only selected parent in matting pool
            crossover_point = np.random.randint(self.NUM_GENES)
            for i in range(0, len(matting_pool), 2):
                parent1 = matting_pool[i]
                parent2 = matting_pool[i+1]
                matting_pool[i] = np.concatenate(
                    (parent1[:crossover_point], parent2[crossover_point:]))
                matting_pool[i+1] = np.concatenate(
                    (parent2[:crossover_point], parent1[crossover_point:]))
                # lakukan pengecekan terhadap kromosom anak hasil penyilangan
                # Tidak boleh ada elemen yang sama
                """
                mt[i] = [a,a,v,c] tidak boleh karena ada elemen a dua kali
                """
                matting_pool[i] = check_validity_child_chromosome(
                    matting_pool[i])
                matting_pool[i +
                             1] = check_validity_child_chromosome(matting_pool[i+1])

            # replace old population and crossover populations
            counter = 0
            for idx_new_chromosome in indices_matting_pool:
                populations[idx_new_chromosome] = matting_pool[counter]
                counter += 1

            return populations

        # define variables
        best_fitness_tracking = []
        worst_fitness_tracking = []

        # Tahap 1: Initialization
        if self.INITIAL_SOLUTIONS != None:
            populations = np.array(self.INITIAL_SOLUTIONS)
        else:
            populations = [[rd.randint(self.MIN_VAL, self.MAX_VAL) for _ in range(
                self.NUM_GENES)] for _ in range(self.POP_SIZE)]
        fitness_scores = [self.FITNESS_FUNCTION(
            chromosome) for chromosome in populations]

        # iterate through generations (stopping criteria)
        for _ in range(self.MAX_GENERATION):

            # Tahap 2: Operator Penyilangan (N-point crossover)
            crossover_populations = crossover(populations)

            # Tahap 3: Mutasi (one point mutation)
            mutated_populations = mutation(crossover_populations)
            # Tahap 4: Fitness Evaluation
            fitness_scores = [self.FITNESS_FUNCTION(np.round(chromosome).astype(int)) for chromosome in mutated_populations]
            best_fitness_tracking.append(max(fitness_scores))
            worst_fitness_tracking.append(min(fitness_scores))

            # Tahap 5: Selection Individu (Rouelete Wheel Selection)
            total_fitness = sum([fitness for fitness in fitness_scores])
            selection_probs = [
                fitness/total_fitness for fitness in fitness_scores]
            index_selected_kromosom = np.random.choice(
                len(mutated_populations), p=selection_probs, size=len(mutated_populations))
            populations = np.round(np.array(mutated_populations)[
                index_selected_kromosom.astype(int)]).astype(int)
            fitness_scores = np.array(fitness_scores)[
                index_selected_kromosom.astype(int)]

        # Tahap 6: Termination and return best solution
        best_index = np.argmax(fitness_scores)
        # get the best solution
        best_thresholds = np.sort(np.array(populations[best_index]))

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = populations[best_index]
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return populations, best_thresholds
