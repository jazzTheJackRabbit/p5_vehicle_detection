import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog
from scipy.ndimage.measurements import label
import pdb

class ImageProcessor:
    def __init__(self, image):
        if type(image) == str:
            self.image = plt.imread(image)
        else:
            self.image = image

        self.working_image = np.copy(self.image)
        self.result_image = np.copy(self.image)
        self.feature_vector = None

        self.window_scale_sizes = [256, 192, 128]
        self.window_y_cutoff = 0.6

        self.window_objs = []
        self.car_windows = []

        self.heat = None
        self.final_car_windows = []

    def convert_to_color_space(self, color_space="RGB"):
        """
        Convert image into desired color space.
        """
        img = np.copy(self.image)
        if color_space != "RGB":
            COLOR = eval("cv2.COLOR_RGB2{}".format(color_space))
            img = cv2.cvtColor(img, COLOR)
        return img

    def extract_features(self, color_space='RGB', spatial_size=(32, 32),hist_bins=32, hist_range=(0, 256),
                         orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
        """
        Extract features based on color-histograms, spatial-binning and HOG for given image. This method is used only
        in the training phase.
        """
        # Extract features
        self.image = self.convert_to_color_space(color_space=color_space)

        spatial_bin_features = self.extract_bin_spatial(size=spatial_size)
        hist_features = self.extract_color_histogram_features(bins=hist_bins, range=hist_range)

        if hog_channel is not None:
            hog_features = self.extract_hog_features(
                orient=orient, 
                pix_per_cell=pix_per_cell, 
                cell_per_block=cell_per_block,
                hog_channel=hog_channel
            )
            self.feature_vector = np.concatenate((spatial_bin_features, hist_features, hog_features))
        else:
            self.feature_vector = np.concatenate((spatial_bin_features, hist_features))        

        return self.feature_vector

    def extract_color_histogram_features(self, bins=32, range=(0,256)):
        """
        Extract color histogram features.
        """
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
        """
        Extract spatially binned features by resizing the image and flattening the color intensity values.
        """
        # Convert image to new color space (if specified)
        img = np.copy(self.image)       
        
        # Use cv2.resize().ravel() to create the feature vector
        resized_img = cv2.resize(img, size)
        spatial_features = resized_img.ravel()
        
        # Return the feature vector
        return spatial_features

    def extract_hog_features(self, img=None,
                             orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, feature_vector=True):
        """
        Extract features based on Histogram of Oriented Gradients (HOG) for a instance's image or sub-image.
        """
        if img is None:
            img = np.copy(self.image)

        hog_features = []

        if hog_channel=="GRAY":
            img = self.convert_to_color_space(color_space="GRAY")
            channel_features, hog_image = hog(
                    img, 
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    block_norm='L2',
                    visualise=True,
                    feature_vector=feature_vector
                )

            hog_features = channel_features

        elif hog_channel=="ALL":         
            for channel in range(img.shape[2]):
                channel_features, hog_image = hog(
                    img[:,:,channel], 
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block),
                    block_norm='L2',
                    visualise=True,
                    feature_vector=feature_vector
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
                    feature_vector=feature_vector
                )

            hog_features = channel_features

        return hog_features

    def extract_hog_features_for_scoring(self, img=None,
                                         orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                                         feature_vector=False):
        """
        Extract HOG features for the prediction phase where the hog-matrix is returned in its original dimensions and
        not un-raveled into a single dimensional feature-vector.
        """

        if img is None:
            img = np.copy(self.image)

        hog_features = hog(
            img[:,:,hog_channel], 
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block),
            block_norm='L2',
            visualise=False,
            feature_vector=feature_vector
        )

        return hog_features

    def find_cars(self,
                  ystart=None, ystop=None,
                  scale=None,
                  classifier=None, scaler=None,
                  color_space='RGB',
                  spatial_size=(32, 32),
                  hist_bins=32, hist_range=(0, 256),
                  orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
        """
        Find cars in the image by converting the raw intensity values of each color channel of the image into feature
        vectors and performing the vehicle detection on small patches of the image (sliding window technique) by using
        the trained classifier and preprocessed StandardScaler from the training phase.
        """
        draw_img = np.copy(self.result_image)
        img = self.image.astype(np.float32)/255
        
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = cv2.cvtColor(img_tosearch,cv2.COLOR_RGB2YCrCb)
        # ctrans_tosearch = img_tosearch
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
            
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
        nfeat_per_block = orient*cell_per_block**2
        
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step

        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = self.extract_hog_features_for_scoring(img=ctrans_tosearch, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=0, feature_vector=False)
        hog2 = self.extract_hog_features_for_scoring(img=ctrans_tosearch, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=1, feature_vector=False)
        hog3 = self.extract_hog_features_for_scoring(img=ctrans_tosearch, orient=orient, pix_per_cell=pix_per_cell,
                                                     cell_per_block=cell_per_block, hog_channel=2, feature_vector=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                
                subimg_processor = ImageProcessor(subimg)

                # Get color features
                spatial_features = subimg_processor.extract_bin_spatial(size=spatial_size)
                hist_features = subimg_processor.extract_color_histogram_features(bins=hist_bins, range=hist_range)

                # Scale features and make a prediction
                test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = classifier.predict(test_features)
                
                if test_prediction == 1:
                    # plt.imshow(subimg)
                    # plt.show()
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    self.car_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 

        self.result_image = draw_img                    
        return draw_img

    def run_find_cars(self, classifier, scaler, previous_frame_processor=None):
        """
        Bootstrap method to run the vehicle detection pipeline for the new image.
        """
        scales = [0.75,1.5,2,3,4]

        window_size = 64
        y_top = 400#int(self.image.shape[0] * 0.6)
        y_bottom = self.image.shape[0]

        for scale in scales:
            out_img = self.find_cars(
                ystart=y_top,
                ystop=y_bottom, 
                scale=scale, 
                classifier=classifier, scaler=scaler
            )

            self.working_image = out_img

        self.add_heat()
        self.apply_threshold(2)

        self.super_impose_previous_frame_detections(previous_frame_processor)
        self.apply_threshold(2)
        
        labels = label(self.heat)

        def draw_labeled_bboxes(img, labels):
            # Iterate through all detected cars
            for car_number in range(1, labels[1]+1):
                # Find pixels with each car_number label value
                nonzero = (labels[0] == car_number).nonzero()
                # Identify x and y values of those pixels
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])

                padding = 10

                # Define a bounding box based on min/max x and y
                left = 0
                if np.min(nonzerox) > padding:
                    left = np.min(nonzerox) - padding
                top = 0
                if np.min(nonzeroy) > padding:
                    top = np.min(nonzeroy) - padding
                right = self.image.shape[1]
                if np.max(nonzerox) < self.image.shape[1]:
                    right = np.max(nonzerox) + padding
                bottom = self.image.shape[0]
                if np.max(nonzeroy) < self.image.shape[0]:
                    bottom= np.max(nonzeroy) + padding

                bbox = ((left, top), (right, bottom))
                self.final_car_windows.append(bbox)

                # Draw the box on the image
                cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

            # Return the image
            return img

        # Draw bounding boxes on a copy of the image
        self.result_image = draw_labeled_bboxes(np.copy(self.image), labels)

    def add_heat(self):
        """
        Create a heat map based on the overlapping detections based on different sliding-window scale sizes.
        :return:
        """
        # Iterate through list of bboxes
        self.heat = np.zeros((self.image.shape[0], self.image.shape[1])).astype(np.float)
        for box in self.car_windows:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        for box in self.car_windows:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] = np.max(self.heat[box[0][1]:box[1][1], box[0][0]:box[1][0]])
        
    def apply_threshold(self, threshold):
        """
        Method to reject false positive detections of vehicles based on thresholding. Select values from the heat map
        only if they are above a certain threshold value.
        """
        # Zero out pixels below the threshold
        self.heat[self.heat <= threshold] = 0

    def super_impose_previous_frame_detections(self, previous_frame_processor=None):
        """
        Use the image processor object from the previous frame to create a more robust detection for the new
        frame.
        """
        # Remove the influence of the previous lingering frames
        if previous_frame_processor and len(previous_frame_processor.heat) > 0:
            previous_frame_processor.heat[previous_frame_processor.heat > 0] -= 1

            # Set the previous frame's heat map to 1
            previous_frame_processor.heat[previous_frame_processor.heat > 0] = 1

            # Super impose the previous frame's heat map to the new frame
            self.heat[previous_frame_processor.heat > 0] += 1

    # ***************************************************************************************************************
    # DEPRECARTED - START
    # ...........
    def draw_boxes(self, bboxes, color=(0, 0, 255), thick=6):
        """
        Draw bounding boxes for an array of box positions on the image processor's original image.
        """
        # Make a copy of the image
        imcopy = self.working_image
        
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

        # Return the image copy with boxes drawn
        return imcopy

    def slide_window(self, x_start_stop=(None, None), y_start_stop=(None, None),
                     xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        """
        Compute rectangular window coordinates for each patch (size defined by parameters) of the image
        """
        x_stop = x_start_stop[1]
        x_search_subset = (x_start_stop[1] - x_start_stop[0])
        x_start = x_stop  - (int(x_search_subset/xy_window[0]) * xy_window[0])
        
        y_stop = y_start_stop[1]
        y_search_subset = (y_start_stop[1] - y_start_stop[0])
        y_start = y_stop - (int(y_search_subset/xy_window[1]) * xy_window[0])
        

        window_list = []
        for j in range(y_stop, y_start, -1*int(xy_window[1]*xy_overlap[1])):
            for i in range(x_stop, x_start, -1*int(xy_window[0]*xy_overlap[0])):
                if (i-xy_window[0]) >= 0 and (j-xy_window[1]) >= 0:
                    window_list.append(
                        (
                            (i-xy_window[0], j-xy_window[1]), (i, j)
                        )
                    )

        self.window_objs.append({
            'window_list': window_list,
            'window_scale_size': xy_window
        })

    def search_windows(self,
            clf=None, scaler=None,
            color_space='RGB',
            spatial_size=(32, 32),
            hist_bins=32, hist_range=(0, 256),
            orient=9, pix_per_cell=8, cell_per_block=2, hog_channel="ALL"
        ):
        img = self.image
        # import pdb; pdb.set_trace();

        #1) Create an empty list to receive positive detection windows
        on_windows = []

        #2) Iterate over all windows in the list
        for window_dict in self.window_objs:
            windows = window_dict['window_list']
            window_scale_size = window_dict['window_scale_size']

            #3) Extract the test window from original image
            for window in windows:
                # test_img = np.zeros((64,64,3))

                try:
                    subimg = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]

                    test_img = cv2.resize(subimg,(64,64))
                except:
                    plt.imshow(subimg); plt.show()
                #4) Extract features for that window using single_img_features()

                test_ip = ImageProcessor(test_img)

                # import pdb; pdb.set_trace();
                features = test_ip.extract_features(
                    color_space=color_space,
                    spatial_size=spatial_size,
                    hist_bins=hist_bins,
                    hist_range=hist_range,
                    orient=orient,
                    pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block,
                    hog_channel=hog_channel
                )

                #5) Scale extracted features to be fed to classifier
                test_features = scaler.transform(np.array(features).reshape(1, -1))

                #6) Predict using your classifier
                prediction = clf.predict(test_features)

                #7) If positive (prediction == 1) then save the window
                if prediction == 1:
                    on_windows.append(window)

        #8) Return windows for positive detections
        self.car_windows = on_windows
        return on_windows

    def find_cars_slow(self, classifier, scaler):
        image = self.image
        car_windows = []
        for window_scale_size in self.window_scale_sizes:
            print("Window size: {}".format(window_scale_size))
            self.slide_window(
                x_start_stop=(0, image.shape[1]),
                y_start_stop=(int(image.shape[0] * self.window_y_cutoff), image.shape[0]),
                xy_window=(window_scale_size, window_scale_size),
                xy_overlap=(0.5, 0.5)
            )

            car_windows.extend(self.search_windows(clf=classifier, scaler=scaler, hog_channel="ALL"))

        self.result_image = self.draw_boxes(car_windows, color=(0, 0, 255), thick=6)
        plt.imshow(self.result_image)
        plt.show()
    # ...........
    # DEPRECARTED - END
    # ***************************************************************************************************************

