import cv2
import numpy as np


def apply_feature_matching(descriptors_img1, descriptors_img2, match_calculator): 
    
    # returns list
    #  match_calculator: whether (calculate_ssd (Sum Square Distance)) or (calculate_ncc (Normalized Cross Correlation))

    # number of key points in each image
    num_key_points_img1 = descriptors_img1.shape[0]
    num_key_points_img2 = descriptors_img2.shape[0]
    # list to store the matches scores
    matches = []

    # loop over each key point in image1
    for kp1 in range(num_key_points_img1):
        # Initial variables which will be updated in the loop
        distance = -np.inf
        y_index = -1

        # calculate similarity with each key point in image2
        # Loop over each key point in image2
        for kp2 in range(num_key_points_img2):

            # Match features between the 2 vectors
            value = match_calculator(descriptors_img1[kp1], descriptors_img2[kp2])

            if value > distance:
                distance = value
                y_index = kp2

        # Create a cv2.DMatch object for each match and set attributes:
        match = cv2.DMatch()
        match.queryIdx = kp1           # queryIdx: The index of the feature in the first image
        match.trainIdx = y_index       # trainIdx: The index of the feature in the second image
        match.distance = distance      # distance: The distance between the two features
        matches.append(match)

    return matches


def calculate_ssd(descriptors_img1, descriptors_img2) : # returns float

    sum_square = 0
    # Get SSD between the 2 vectors
    for m in range(len(descriptors_img1)):
        sum_square += (descriptors_img1[m] - descriptors_img2[m]) ** 2

    sum_square = - (np.sqrt(sum_square))  # negative because it will be reversed after that

    return sum_square

def calculate_ncc(descriptors_img1, descriptors_img2) : # returns float
    
    # Normalize the 2 vectors
    out1_normalized = (descriptors_img1 - np.mean(descriptors_img1)) / (np.std(descriptors_img1))
    out2_normalized = (descriptors_img2 - np.mean(descriptors_img2)) / (np.std(descriptors_img2))

    # Apply similarity product between the 2 normalized vectors
    correlation_vector = np.multiply(out1_normalized, out2_normalized)

    # Get mean of the result vector
    correlation = float(np.mean(correlation_vector))

    return correlation
