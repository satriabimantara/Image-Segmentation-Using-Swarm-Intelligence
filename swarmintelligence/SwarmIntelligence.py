import numpy as np
import math


class SwarmIntelligence:
    def __init__(self, *args, **kwargs):
        self.class_name = 'SwarmIntelligence'
        if kwargs['class_name'] is not None:
            self.class_name = kwargs['class_name']
        # set initial solution
        self.INITIAL_SOLUTIONS = None
        if not (kwargs['initial_solution'] is None):
            self.INITIAL_SOLUTIONS = kwargs['initial_solution']

        # set class properties variables after training phase was done
        self.PARAMS_TRAINING = False
        self.BEST_SOLUTION = None
        self.BEST_IDX_SOLUTION = None
        self.BEST_FITNESS_TRACKING = None
        self.WORST_FITNESS_TRACKING = None

    def fit_run(self, image_array):
        # initialize image input properties
        self.PIXEL_FREQ = np.bincount(image_array.ravel(), minlength=256)
        self.TOTAL_PIXELS = np.sum(self.PIXEL_FREQ)
        self.PROB_PIXEL_INTENSITY = [
            pixel_freq/self.TOTAL_PIXELS for pixel_freq in self.PIXEL_FREQ]
        return self

    def get_params_training_(self):
        if self.PARAMS_TRAINING == False:
            return None
        else:
            return {
                'best_solution': self.BEST_SOLUTION,
                'best_fitness_tracking': self.BEST_FITNESS_TRACKING,
                'worst_fitness_tracking': self.WORST_FITNESS_TRACKING
            }
    # APPLY LIST OF OBJECTIVE FUNCTION HERE

    def otsu_method(self, thresholds):
        probabilities_pixel_intensity = self.PROB_PIXEL_INTENSITY
        batas_bawah_threshold = None
        batas_atas_threshold = None
        thresholds = np.sort(thresholds)

        # variabel untuk menampung omega_i, miu_i, miu_T
        omega = []
        miu_i = []
        miu_T = 0
        for idx_region in range(len(thresholds)+1):
            if idx_region == 0:
                batas_bawah_threshold = 0
                batas_atas_threshold = thresholds[idx_region]-1
            elif idx_region == len(thresholds):
                batas_bawah_threshold = thresholds[idx_region-1]
                batas_atas_threshold = 254
            else:
                batas_bawah_threshold = thresholds[idx_region-1]
                batas_atas_threshold = thresholds[idx_region]-1

            # hitung omega_i (Compute the cumulative sums of probabilities up to i)
            omega_region_i = 0
            for i in range(batas_bawah_threshold, batas_atas_threshold+1):
                omega_region_i += probabilities_pixel_intensity[i]
            omega.append(omega_region_i)

            # hitung miu_i (Mean intensity value of region i)
            mean_intensity_region_i = 0
            # laplace smoothing for anticipating dividing by zero
            laplace = 0
            if omega[idx_region] == 0:
                # get the epsilon value aka nilai positive number terkecil dari mesin
                laplace = np.finfo(float).eps
            for i in range(batas_bawah_threshold, batas_atas_threshold+1):
                mean_intensity_region_i += (i*probabilities_pixel_intensity[i])/(
                    omega[idx_region] + laplace)
            miu_i.append(mean_intensity_region_i)

            # hitung miu_t
            miu_T += omega[idx_region] * miu_i[idx_region]

        # calculate otsu criteria
        sigma_b = 0
        for idx_region in range(len(thresholds)+1):
            sigma_b += omega[idx_region] * \
                math.pow((miu_i[idx_region] - miu_T), 2)
        fitness = sigma_b
        return fitness

    def kapur_entropy_method(self, thresholds):
        probabilities_pixel_intensity = self.PROB_PIXEL_INTENSITY
        batas_bawah_threshold = None
        batas_atas_threshold = None

        # gene representation
        thresholds = np.sort(np.array(thresholds))

        # variabel untuk menampung omega_i, miu_i, miu_T
        omega = list()
        entropy_i = list()

        for idx_region in range(len(thresholds)+1):
            if idx_region == 0:
                batas_bawah_threshold = 0
                batas_atas_threshold = thresholds[idx_region]-1
            elif idx_region == len(thresholds):
                batas_bawah_threshold = thresholds[idx_region-1]
                batas_atas_threshold = 254
            else:
                batas_bawah_threshold = thresholds[idx_region-1]
                batas_atas_threshold = thresholds[idx_region]-1

            # hitung omega_i (Compute the cumulative sums of probabilities up to i)
            omega_region_i = 0
            for i in range(batas_bawah_threshold, batas_atas_threshold+1):
                omega_region_i += probabilities_pixel_intensity[i]
            omega.append(omega_region_i)

            # laplace smoothing for anticipating dividing by zero
            laplace = 0
            if omega[idx_region] == 0:
                laplace = np.finfo(float).eps

            # hitung entropy region ke-i (Ei)
            entropy_region_i = 0
            for i in range(batas_bawah_threshold, batas_atas_threshold+1):
                x = (
                    probabilities_pixel_intensity[i]/(omega[idx_region]+laplace))
                if x != 0:
                    entropy_region_i += -1 * (x * np.log(x))

            entropy_i.append(entropy_region_i)

        # calculate fitness value (summing all element in entropy_i)
        fitness = np.sum(np.array(entropy_i))
        return fitness

    def mMasi_entropy_method(self, thresholds, alpha=0.87):
        probabilities_pixel_intensity = self.PROB_PIXEL_INTENSITY
        batas_bawah_threshold = None
        batas_atas_threshold = None

        # gene representation
        thresholds = np.sort(np.array(thresholds))
        # variabel untuk menampung hasil perhitungan
        omega = []
        phi = []
        MME = []
        for idx_region in range(len(thresholds)+1):
            # tentukan batas bawah dan batas atas dari setiap region yang dibatasi dua threshold
            if idx_region == 0:
                batas_bawah_threshold = 0
                batas_atas_threshold = thresholds[idx_region]-1
            elif idx_region == len(thresholds):
                batas_bawah_threshold = thresholds[idx_region-1]
                batas_atas_threshold = 254
            else:
                batas_bawah_threshold = thresholds[idx_region-1]
                batas_atas_threshold = thresholds[idx_region]-1

            # hitung omega_i (Compute the cumulative sums of probabilities up to i (Pi))
            omega_region_i = 0
            for i in range(batas_bawah_threshold, batas_atas_threshold+1):
                omega_region_i += probabilities_pixel_intensity[i]
            omega.append(omega_region_i)

            # calculate phi_i (cumulative sum of (Pi * log2(Pi)))
            phi_region_i = 0
            laplace = 0
            if omega_region_i == 0:
                laplace = np.finfo(float).eps
            for i in range(batas_bawah_threshold, batas_atas_threshold+1):
                if probabilities_pixel_intensity[i] != 0:
                    value = (
                        probabilities_pixel_intensity[i] / omega_region_i+laplace)
                    phi_region_i += value * math.log2(value)
            phi.append(phi_region_i)

            # calculate MME for each region
            value = 1 - (1-alpha)*phi_region_i
            mme_region_i = math.log2(value)/(1-alpha)
            MME.append(mme_region_i)

        # calculate M.Masi Entropy criteria
        sigma_MME = np.sum(np.array(MME))
        fitness = sigma_MME
        return fitness
