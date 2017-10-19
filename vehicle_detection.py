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

from pipeline import ImageProcessor

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

        self.current_frame_processor = None
        self.previous_frame_processor = None

    def extract_features_for_images(self, images, 
        color_space='RGB', 
        spatial_size=(32, 32),
        hist_bins=32, hist_range=(0, 256), 
        orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
        # Create a list to append feature vectors to
        print(self.extract_features_for_images.__name__)
        feature_vectors = []

        for i,image in enumerate(images):
            # Read in each one by one
            image_processor = ImageProcessor(image)
            image_processor.extract_features(
                color_space=color_space, 
                spatial_size=spatial_size, 
                hist_bins=hist_bins, 
                hist_range=hist_range,
                orient=orient, 
                pix_per_cell=pix_per_cell, 
                cell_per_block=cell_per_block,
                hog_channel=hog_channel
            )
            feature_vectors.append(image_processor.feature_vector)
            if (i%100 == 0):
                print("{}/{}".format(str(i),str(len(images))))
            
        self.feature_vectors = feature_vectors

        # Return list of feature vectors
        return feature_vectors

    def scale_feature_vectors(self, car_features, notcar_features):
        print(self.scale_feature_vectors.__name__)
        scaled_X = None

        if len(car_features) > 0:
            # y stack of feature vectors
            X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)

            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)

            self.scaler = X_scaler

        return scaled_X

    def extract_image_paths(self, path=None):
        print(self.extract_image_paths.__name__)
        cars = []
        notcars = []

        if path is not None:
            images = glob.glob(path)

            # import pdb; pdb.set_trace();
            for image in images:
                if 'non-vehicles' in image:
                    notcars.append(image)
                else:
                    cars.append(image)
        print(len(cars),len(notcars))
        return cars, notcars

    def preprocess_data(self, path):
        print(self.preprocess_data.__name__)
        cars, notcars = self.extract_image_paths(path=path)
        
        # Feature extraction for training        
        car_features = self.extract_features_for_images(cars, 
            color_space='YCrCb', spatial_size=(32, 32),
            hist_bins=32, hist_range=(0, 256),
            hog_channel="ALL"
        )
        
        notcar_features = self.extract_features_for_images(notcars, 
            color_space='YCrCb', spatial_size=(32, 32),
            hist_bins=32, hist_range=(0, 256),
            hog_channel="ALL"
        )

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
        print(self.train.__name__)
        # Use a linear SVC (support vector classifier)
        self.classifier = LinearSVC()

        # Train the SVC
        self.classifier.fit(self.X_train, self.y_train)
        print('Test Accuracy of SVC = ', self.classifier.score(self.X_test, self.y_test))

    def find_vehicles(self, image):
        self.current_frame_processor = ImageProcessor(image)

        if self.previous_frame_processor is None:            
            self.current_frame_processor.run_find_cars(self.classifier, self.scaler)
        else:
            self.current_frame_processor.run_find_cars(self.classifier, self.scaler, previous_frame_processor=self.previous_frame_processor)

        self.previous_frame_processor = self.current_frame_processor

        return self.current_frame_processor.result_image
