class Utilization:
    def __init__(self):
        pass

    def digitize(self, image_array, thresholds):
        # image dimension
        row = image_array.shape[0]
        col = image_array.shape[1]
        regions_threshold = dict()
        # flatten image to 1D
        x = image_array.copy().ravel()

        # get lower and upper bound
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
            regions_threshold[idx_region] = [
                batas_bawah_threshold, batas_atas_threshold]

        # convert p(i,j) to correpondent lower and upper bound
        """
        bb <= p(i,j) <=ba --> ba+1
        """
        for idx_x, y in enumerate(x):
            for region_id, interval in regions_threshold.items():
                if y >= interval[0] and y <= interval[1]:
                    x[idx_x] = interval[1]+1
                    break

        # convert back segmented image into their original shape
        return x.reshape((row, col))
