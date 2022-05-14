import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class MeanShift:
    def __init__(self, source: np.ndarray, threshold: int):
        self.threshold = threshold
        self.current_mean_random = True
        self.current_mean_arr = []

        # Output Array
        size = source.shape[0], source.shape[1], 3
        self.output_array = np.zeros(size, dtype=np.uint8)

        # Create Feature Space
        self.feature_space = self.create_feature_space(source=source)

    def run_mean_shift(self):
        while len(self.feature_space) > 0:
            below_threshold_arr, self.current_mean_arr = self.calculate_euclidean_distance(
                current_mean_random=self.current_mean_random,
                threshold=self.threshold)

            self.get_new_mean(below_threshold_arr=below_threshold_arr)

    def get_output(self):
        return self.output_array

    @staticmethod
    def create_feature_space(source: np.ndarray):

        row = source.shape[0]
        col = source.shape[1]

        # Feature Space Array
        feature_space = np.zeros((row * col, 5))

        # Array to store RGB Values for each pixel
        # rgb_array = np.array((1, 3))

        counter = 0

        for i in range(row):
            for j in range(col):
                rgb_array = source[i][j]

                for k in range(5):
                    if (k >= 0) & (k <= 2):
                        feature_space[counter][k] = rgb_array[k]
                    else:
                        if k == 3:
                            feature_space[counter][k] = i
                        else:
                            feature_space[counter][k] = j
                counter += 1

        return feature_space

    def calculate_euclidean_distance(self, current_mean_random: bool, threshold: int):
        """
        calculate the Euclidean distance of all the other pixels in M with the current mean.
        """

        below_threshold_arr = []

        # selecting a random row from the feature space and assigning it as the current mean
        if current_mean_random:
            current_mean = np.random.randint(0, len(self.feature_space))
            self.current_mean_arr = self.feature_space[current_mean]

        for f_indx, feature in enumerate(self.feature_space):
            # Finding the euclidean distance of the randomly selected row i.e. current mean with all the other rows
            ecl_dist = euclidean_distance(self.current_mean_arr, feature)

            # Checking if the distance calculated is within the threshold. If yes taking those rows and adding
            # them to a list below_threshold_arr
            if ecl_dist < threshold:
                below_threshold_arr.append(f_indx)

        return below_threshold_arr, self.current_mean_arr

    def get_new_mean(self, below_threshold_arr: list):
        iteration = 0.01

        # For all the rows found and placed in below_threshold_arr list, calculating the average of
        # Red, Green, Blue and index positions.
        print(f"features shape: {self.feature_space.shape}")
        mean_r = np.mean(self.feature_space[below_threshold_arr][:, 0])
        mean_g = np.mean(self.feature_space[below_threshold_arr][:, 1])
        mean_b = np.mean(self.feature_space[below_threshold_arr][:, 2])
        mean_i = np.mean(self.feature_space[below_threshold_arr][:, 3])
        mean_j = np.mean(self.feature_space[below_threshold_arr][:, 4])

        # Finding the distance of these average values with the current mean and comparing it with iter
        mean_e_distance = (euclidean_distance(mean_r, self.current_mean_arr[0]) +
                           euclidean_distance(mean_g, self.current_mean_arr[1]) +
                           euclidean_distance(mean_b, self.current_mean_arr[2]) +
                           euclidean_distance(mean_i, self.current_mean_arr[3]) +
                           euclidean_distance(mean_j, self.current_mean_arr[4]))

        # If less than iter, find the row in below_threshold_arr that has i, j nearest to mean_i and mean_j
        # This is because mean_i and mean_j could be decimal values which do not correspond
        # to actual pixel in the Image array.
        if mean_e_distance < iteration:
            new_arr = np.zeros((1, 3))
            new_arr[0][0] = mean_r
            new_arr[0][1] = mean_g
            new_arr[0][2] = mean_b

            # When found, color all the rows in below_threshold_arr with
            # the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
            for i in range(len(below_threshold_arr)):
                m = int(self.feature_space[below_threshold_arr[i]][3])
                n = int(self.feature_space[below_threshold_arr[i]][4])
                self.output_array[m][n] = new_arr

                # Also now don't use those rows that have been colored once.
                self.feature_space[below_threshold_arr[i]][0] = -1

            self.current_mean_random = True
            new_d = np.zeros((len(self.feature_space), 5))
            counter_i = 0

            for i in range(len(self.feature_space)):
                if self.feature_space[i][0] != -1:
                    new_d[counter_i][0] = self.feature_space[i][0]
                    new_d[counter_i][1] = self.feature_space[i][1]
                    new_d[counter_i][2] = self.feature_space[i][2]
                    new_d[counter_i][3] = self.feature_space[i][3]
                    new_d[counter_i][4] = self.feature_space[i][4]
                    counter_i += 1

            self.feature_space = np.zeros((counter_i, 5))

            counter_i -= 1
            for i in range(counter_i):
                self.feature_space[i][0] = new_d[i][0]
                self.feature_space[i][1] = new_d[i][1]
                self.feature_space[i][2] = new_d[i][2]
                self.feature_space[i][3] = new_d[i][3]
                self.feature_space[i][4] = new_d[i][4]

        else:
            self.current_mean_random = False
            self.current_mean_arr[0] = mean_r
            self.current_mean_arr[1] = mean_g
            self.current_mean_arr[2] = mean_b
            self.current_mean_arr[3] = mean_i
            self.current_mean_arr[4] = mean_j


def apply_mean_shift(source: np.ndarray, threshold: int = 60):
    src = np.copy(source)
    ms = MeanShift(source=src, threshold=threshold)
    ms.run_mean_shift()
    output = ms.get_output()

    return output


if __name__ == "__main__":
    
    img = cv2.imread('../images/landscape.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
    
    start_time = time.time()
    segmented_image = apply_mean_shift(img_luv)    
    end_time = time.time()

    mean_shift_time = format(end_time - start_time, '.5f')
    print(f'Mean Shift Computation Time = {mean_shift_time} sec')    # print(  end_time - start_time)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))    
    ax = axes.ravel()

    ax[0].imshow(img_rgb)
    ax[0].set_title('Original Image')
    ax[0].set_axis_off()

    ax[1].imshow(segmented_image )
    ax[1].set_title('Luv Image')
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()
