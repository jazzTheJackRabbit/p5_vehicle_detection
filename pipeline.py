import cv2
import numpy as np
import cv2
import matplotlib.image as mpimg

from os.path import join, realpath, abspath
from matplotlib import pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split

class VehicleDetectionPipeline:
	def __init__(self):
		pass

	def train(self):
		pass

	def find_vehicles(self, video_path):
		pass

class ImageFeaturizer:
	def __init__(self, image):
		if type(image) == str:
			self.image = plt.imread(image)
		else:
			self.image = image

		self.feature_vector = None

	def color_histogram_features(self, bins=32, range=(0,256)):
	    # Compute the histogram of the RGB channels separately
	    rhist = np.histogram(self.image[:,:,0], bins=bins, range=range)
	    ghist = np.histogram(self.image[:,:,1], bins=bins, range=range)
	    bhist = np.histogram(self.image[:,:,2], bins=bins, range=range)

	    # Generating bin centers
	    bin_edges = rhist[1]
	    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
	    # Concatenate the histograms into a single feature vector
	    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
	    # Return the individual histograms, bin_centers and feature vector
	    return rhist, ghist, bhist, bin_centers, hist_features