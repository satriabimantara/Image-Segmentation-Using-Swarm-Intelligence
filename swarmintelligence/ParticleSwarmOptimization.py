from .SwarmIntelligence import SwarmIntelligence
import random as rd
import numpy as np


class ParticleSwarmOptimization(SwarmIntelligence):
    def __init__(self, k, particleSize, maxIteration, phi1, phi2, inertia, fitness_function='otsu', initial_solution=None):
        super(ParticleSwarmOptimization, self).__init__(
            class_name='ParticleSwarmOptimization', initial_solution=initial_solution)
        # initialize PSO parameter
        self.NUM_PARTICLE_ELEMENT = k
        self.PARTICLE_SIZE = particleSize
        self.MAX_ITERATION = maxIteration
        self.PHI1 = phi1
        self.PHI2 = phi2
        self.INERTIA = inertia

        # set objective function
        self.FITNESS_FUNCTION = self.otsu_method
        if fitness_function == 'kapur_entropy':
            self.FITNESS_FUNCTION = self.kapur_entropy_method
        elif fitness_function == 'm_masi_entropy':
            self.FITNESS_FUNCTION = self.mMasi_entropy_method

    def fit_run(self, image_array):
        super(ParticleSwarmOptimization, self).fit_run(image_array)
        # tentukan Batas atas dan Batas bawah dari partikel
        self.LOWER_BOUND = min(image_array.ravel())
        self.UPPER_BOUND = max(image_array.ravel())
        # tentukan kecepatan maksimum dan minimum dari partikel
        self.V_MIN = self.LOWER_BOUND
        self.V_MAX = self.UPPER_BOUND

        # run particle swarm optimization process
        return self.pso()

    def pso(self):
        """
        TAHAP 1: Inisialisasi
        - Inisialisasi Matriks Partikel (nxm): n Partikel, m threshold
        - Inisialisasi kecepatan awal partikel dengan 0
        - Inisialisasi fitness partikel dengan objektif function
        - Inisialisasi PBest dan GBest partikel 
        - Inisialisasi variabel best fitness dan best worst tracking
        """

        # inisialisasi vektor kecepatan dengan 0
        velocities = np.zeros((self.PARTICLE_SIZE, self.NUM_PARTICLE_ELEMENT))
        if self.INITIAL_SOLUTIONS != None:
            particles = np.array(self.INITIAL_SOLUTIONS)
        else:
            # inisialisasi partikel 0 - 255 random
            particles = np.random.randint(self.LOWER_BOUND, self.UPPER_BOUND,  size=(
                self.PARTICLE_SIZE, self.NUM_PARTICLE_ELEMENT))

        # inisialisasi fitness (best values) setiap partikel PBest
        pBestposition = particles
        pBestValue = [float('-inf') for i in range(self.PARTICLE_SIZE)]

        # inisialisasi Global Best (GBest)
        global_best_position = np.zeros(self.NUM_PARTICLE_ELEMENT)
        global_best_value = float("-inf")

        # inisialisasi best fitness dan worst fitness tracking
        best_fitness_tracking = list()
        worst_fitness_tracking = list()

        """
        Tahap 2: Optimasi dengan PSO
        - Iterasi sampai maximum iterasi
        - Perbarui vektor kecepatan
        - Perbarui posisi
        - Perbarui PBest dan GBest
        """
        w = self.INERTIA
        c1 = self.PHI1
        c2 = self.PHI2

        for iteration in range(self.MAX_ITERATION):
            # Perbarui vektor kecepatan untuk semua partikel
            r1 = np.random.rand(self.PARTICLE_SIZE,
                                self.NUM_PARTICLE_ELEMENT)
            r2 = np.random.rand(self.PARTICLE_SIZE,
                                self.NUM_PARTICLE_ELEMENT)
            velocities = w*velocities + c1*r1*(
                pBestposition - particles) + c2*r2*(global_best_position - particles)

            # Update swarm position
            particles = particles + np.round(velocities).astype('int64')

            # Apply boundaries to the particles new position
            for idx_particle, particle in enumerate(particles):
                particles[idx_particle, particles[idx_particle, :] <
                          self.LOWER_BOUND] = self.LOWER_BOUND
                particles[idx_particle, particles[idx_particle, :] >
                          self.UPPER_BOUND] = self.UPPER_BOUND - (round(self.INERTIA * np.random.randint(self.LOWER_BOUND, self.UPPER_BOUND)))
                velocities[idx_particle] = np.clip(
                    velocities[idx_particle], self.V_MIN, self.V_MIN)

            # Evaluate fitness each particle
            fitness_particle_now = [self.FITNESS_FUNCTION(
                particle) for particle in particles]

            # Update PBest and GBest
            for idx_particle in range(self.PARTICLE_SIZE):
                if fitness_particle_now[idx_particle] > pBestValue[idx_particle]:
                    pBestposition[idx_particle] = particles[idx_particle]
                    pBestValue[idx_particle] = fitness_particle_now[idx_particle]
            max_index = np.argmax(pBestValue)
            if pBestValue[max_index] > global_best_value:
                global_best_position = pBestposition[max_index]
                global_best_value = pBestValue[max_index]

            best_fitness_tracking.append(global_best_value)
            worst_fitness_tracking.append(min(pBestValue))

        # cari partikel yang fitness nya paling optimal dari GBest position dan Fitness
        best_index = np.argmax(pBestValue)
        best_thresholds = np.sort(np.array(global_best_position))

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = True
        self.BEST_SOLUTION = best_thresholds
        self.BEST_IDX_SOLUTION = best_index
        self.BEST_FITNESS_TRACKING = best_fitness_tracking
        self.WORST_FITNESS_TRACKING = worst_fitness_tracking

        return particles, best_thresholds
