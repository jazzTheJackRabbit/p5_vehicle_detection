import cv2
import glob
import matplotlib.image as mpimg
import numpy as np

from matplotlib import pyplot as plt
from os.path import abspath
from os.path import join
from os.path import realpath
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class VehicleDetectionPipeline:
    def __init__(self):
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.feature_vectors = []
        self.scaler = None

        self.classifier = None

    def extract_features_for_images(self, images, cspace='RGB', spatial_size=(32, 32),
        hist_bins=32, hist_range=(0, 256)):
        # Create a list to append feature vectors to
        feature_vectors = []

        for image in images:
            # Read in each one by one
            image_processor = ImageProcessor(image)
            image_processor.extract_features(
                color_space=color_space, 
                spatial_size=spatial_size, 
                hist_bins=hist_bins, 
                hist_range=hist_range
            )
            feature_vectors.append(image_processor.feature_vector)
            
        self.feature_vectors = feature_vectors

        # Return list of feature vectors
        return feature_vectors

    def scale_feature_vectors(self, car_features, notcar_features):
        scaled_X = None

        if len(car_features) > 0:
            # y stack of feature vectors
            X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)

            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)
            car_ind = np.random.randint(0, len(cars))

            self.scaler = X_scaler

        return scaled_X

    def extract_image_paths(self, path=None):
        cars = []
        notcars = []

        if path is not None:
            images = glob.glob(path)

            for image in images:
                if 'image' in image or 'extra' in image:
                    notcars.append(image)
                else:
                    cars.append(image)

        return cars, notcars

    def preprocess_data(self):
        cars, notcars = self.extract_image_paths(path='../data/vehicle_detection/*/*1/*.jpeg')
                
        car_features = self.extract_features_for_images(cars, cspace='RGB', spatial_size=(32, 32),
                                hist_bins=32, hist_range=(0, 256))
        
        notcar_features = self.extract_features_for_images(notcars, cspace='RGB', spatial_size=(32, 32),
                                hist_bins=32, hist_range=(0, 256))

        self.X = self.scale_feature_vectors(car_features, notcar_features)
        self.y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        rand_state = np.random.randint(0, 100)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size=0.2, 
            random_state=rand_state
        )

    def train(self):
        # Use a linear SVC (support vector classifier)
        self.classifier = LinearSVC()

        # Train the SVC
        self.classifier.fit(self.X_train, self.y_train)
        print('Test Accuracy of SVC = ', svc.score(self.X_test, self.y_test))

    def find_vehicles(self, video_path):
        pass


class ImageProcessor:
    def __init__(self, image):
        if type(image) == str:
            self.image = plt.imread(image)
        else:
            self.image = image

        self.feature_vector = None

    def convert_to_color_space(self, color_space="RGB"):
        img = np.copy(self.image)
        if color_space != "RGB":
            COLOR = eval("cv2.COLOR_RGB2{}".format(color_space))
            img = cv2.cvtColor(img, COLOR)
        return img

    def extract_features(self, color_space='RGB', spatial_size=(32, 32),
        hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, 
        cell_per_block=2, hog_channel=0,):
        # Extract features
        spatial_bin_features = self.extract_bin_spatial(size=spatial_size)
        hist_features = self.extract_color_histogram_features(bins=hist_bins, range=hist_range)
        hog_features = self.extract_hog_features(
            orient=orient, 
            pix_per_cell=pix_per_cell, 
            cell_per_block=cell_per_block,
            channels=hog_channel
        )
        self.feature_vector = np.concatenate((spatially_binned, hist_features, hog_features))
        return self.feature_vector

    def extract_color_histogram_features(self, bins=32, range=(0,256)):
        # Compute the histogram of the RGB channels separately
        rhist = np.histogram(self.image[:,:,0], bins=bins, range=range)
        ghist = np.histogram(self.image[:,:,1], bins=bins, range=range)
        bhist = np.histogram(self.image[:,:,2], bins=bins, range=range)

        # Generating bin centers
        bin_edges = rhist[1]
        bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

        return hist_features

    def extract_bin_spatial(self, size=(32, 32)):
        # Convert image to new color space (if specified)
        img = np.copy(self.image)       
        
        # Use cv2.resize().ravel() to create the feature vector
        resized_img = cv2.resize(img, size)
        spatial_features = resized_img.ravel()
        
        # Return the feature vector
        return spatial_features

    def extract_hog_features(self, orient=9, pix_per_cell=8, 
        cell_per_block=2, hog_channel="GRAY"):

        img = np.copy(self.image)
        hog_features = []

        if hog_channel=="GRAY":
            img = self.convert_to_color_space(color_space="GRAY")

        elif hog_channel=="ALL":         
            for channel in range(feature_image.shape[2]):
                channel_features, hog_image = hog(
                    img[:,:,channel], 
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    block_norm='L2',
                    visualise=True,
                    feature_vector=True
                )

                hog_features.extend(channel_features)
        else:
            channel_features, hog_image = hog(
                    img[:,:,hog_channel], 
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    block_norm='L2',
                    visualise=True,
                    feature_vector=True
                )

            hog_features = channel_features

        return hog_features

    def draw_boxes(self, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(self.image)
        
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

        # Return the image copy with boxes drawn
        return imcopy

    def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None), 
        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        # If x and/or y start/stop positions not defined, set to image size
        # Compute the span of the region to be searched    
        # Compute the number of pixels per step in x/y
        # Compute the number of windows in x/y
        # Initialize a list to append window positions to
        
        x_stop = x_start_stop[1]
        x_search_subset = (x_start_stop[1] - x_start_stop[0])
        x_start = x_stop  - (int(x_search_subset/xy_window[0]) * xy_window[0])
        
        y_stop = y_start_stop[1]
        y_search_subset = (y_start_stop[1] - y_start_stop[0])    
        y_start = y_stop - (int(y_search_subset/xy_window[1]) * xy_window[0])
        
        window_list = []
        for j in range(y_stop, y_start, -1*int(xy_window[1]*xy_overlap[1])):
            for i in range(x_stop, x_start, -1*int(xy_window[0]*xy_overlap[0])):
                window_list.append(
                    (
                        (i-xy_window[0], j-xy_window[1]), (i, j)
                    )
                )
        return window_list













