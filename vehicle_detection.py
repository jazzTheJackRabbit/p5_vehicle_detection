import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import pdb

from image_processor import ImageProcessor

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

    def load(self, classifier="classifier-all-ycrcb.p", scaler="scaler-all-ycrcb.p"):
        self.X_train = pickle.load(open('saved_features/X_train.p', 'rb'))
        self.X_test= pickle.load(open('saved_features/X_test.p', 'rb'))
        self.y_train = pickle.load(open('saved_features/y_train.p', 'rb'))
        self.y_test = pickle.load(open('saved_features/y_test.p'), 'rb')

        self.classifier = pickle.load(open("saved_models/{}".format(classifier), 'rb'))
        self.scaler = pickle.load(open("saved_models/{}".format(scaler), 'rb'))

    def extract_features_for_images(self, images, color_space='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256), orient=9, pix_per_cell=8, cell_per_block=2,
                                    hog_channel=0):
        """
        Extract features based on color-histograms, spatial-binning and histogram of oriented gradients
        (HOG) and creates feature vectors for all the images in the dataset and creates the training data required
        for training.
        """
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
            if (i%1000 == 0):
                print("{}/{}".format(str(i),str(len(images))))
            
        self.feature_vectors = feature_vectors

        # Return list of feature vectors
        return feature_vectors

    def scale_feature_vectors(self, car_features, not_car_features):
        """
        Scale/Normalize feature vector values for each feature so that none of the feature sets (color-histograms/
        spatially-binned/HOG features) dominate each other.
        """
        print(self.scale_feature_vectors.__name__)
        scaled_X = None

        if len(car_features) > 0:
            # y stack of feature vectors
            X = np.vstack((car_features, not_car_features)).astype(np.float64)

            # Fit a per-column scaler
            X_scaler = StandardScaler().fit(X)

            # Apply the scaler to X
            scaled_X = X_scaler.transform(X)

            self.scaler = X_scaler

        return scaled_X

    def extract_image_paths(self, path=None):
        """
        Helper method to extract image paths for training images.
        """
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

    def pre_process_data(self, path):
        """
        Pre-process the images from the dataset to create feature vectors for each image and create training and
        testing sets of features and labels.
        """
        print(self.pre_process_data.__name__)
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
        """
        Train a Linear Support Vector Classifier on the training dataset of feature vectors previously created.
        """
        print(self.train.__name__)
        # Use a linear SVC (support vector classifier)
        self.classifier = LinearSVC()

        # Train the SVC
        self.classifier.fit(self.X_train, self.y_train)

    def test_performance(self):
        print('Test Accuracy of SVC = ', self.classifier.score(self.X_test, self.y_test))

    def find_vehicles(self, image):
        """
        Perform vehicle detection for a given image, using the trained classifier.
        """
        self.current_frame_processor = ImageProcessor(image)

        if self.previous_frame_processor is None:            
            self.current_frame_processor.run_find_cars(self.classifier, self.scaler)
        else:
            self.current_frame_processor.run_find_cars(self.classifier, self.scaler, previous_frame_processor=self.previous_frame_processor)

        self.previous_frame_processor = self.current_frame_processor

        return self.current_frame_processor.result_image
