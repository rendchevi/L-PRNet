'''
===============================
 L-PRNet Inference Script
 -----------------------------
 Rendi Chevi
 https://github.com/rendchevi
===============================
'''
import tensorflow as tf
import cv2
import dlib
from imutils import face_utils
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import open3d as o3d

from skimage.transform import resize
from skimage.color import rgb2gray
from matplotlib.colors import LinearSegmentedColormap

class LPRNet:
    def __init__(self, resolution = 128):
        self.resolution = resolution
        self.model = tf.keras.models.load_model('model/lprnet_128', compile = False)
        self.face_detector = dlib.cnn_face_detection_model_v1('model/mmod_human_face_detector.dat')
        self.bbox = None

    def normalize_image(self, input_img):
        '''
        INPUT : [1] input_img; RGB image, numpy array
        OUTPUT: [1] input_img; normalized input image
        '''
        # Normalize input image to [0,1]
        input_img[:,0] = (input_img[:,0] - np.min(input_img[:,0])) / (np.max(input_img[:,0]) - np.min(input_img[:,0]))
        input_img[:,1] = (input_img[:,1] - np.min(input_img[:,1])) / (np.max(input_img[:,1]) - np.min(input_img[:,1]))
        input_img[:,2] = (input_img[:,2] - np.min(input_img[:,2])) / (np.max(input_img[:,2]) - np.min(input_img[:,2]))

        return input_img

    def preprocess_image(self, input_img):
        '''
        INPUT : [1] input_img; RGB image, numpy array
        OUTPUT: [1] input_img; Preprocessed image for L-PRNet, tf.Tensor
        '''
        # Normalize input image and scale to [0, 1]
        input_img = self.normalize_image(input_img.copy())
        # Resize input image
        input_img = resize(input_img, (self.resolution, self.resolution))
        # Convert to tensor and expand dimension
        input_img = tf.convert_to_tensor(input_img)
        input_img = tf.expand_dims(input_img, axis = 0)

        return input_img

    def pointcloud_scaling(self, pcl, min_x, max_x, min_y, max_y):
        pcl[:,0] = min_x + (((pcl[:,0] - np.min(pcl[:,0])) * (max_x - min_x)) / (np.max(pcl[:,0]) - np.min(pcl[:,0])))
        pcl[:,1] = min_x + (((pcl[:,1] - np.min(pcl[:,1])) * (max_x - min_x)) / (np.max(pcl[:,1]) - np.min(pcl[:,1])))
        pcl[:,2] = min_x + (((pcl[:,2] - np.min(pcl[:,2])) * (max_x - min_x)) / (np.max(pcl[:,2]) - np.min(pcl[:,2])))

        return pcl

    def predict_uv(self, input_img):
        '''
        INPUT : [1] input_img; RGB image, numpy array
        OUTPUT: [1] uv_map; UV representation of the 3D face, numpy array
        '''
        # Preprocess input image
        input_img = self.preprocess_image(input_img)
        # Feed image to the network
        uv_map = self.model.predict(input_img)
        # Get predicted UV-Map
        uv_map = tf.squeeze(uv_map, axis = 0).numpy()

        return uv_map

    def detect_face(self, input_img, detect_size = 256, scale = 1.15):
        '''
        INPUT : [1] input_img; RGB image, numpy array
                [2] detect_size; resized image for the detector, int
        OUTPUT: [1] cropped_face; face region in the image
        '''
        # Normalize input image
        input_img_proc = self.normalize_image(input_img.copy())
        # Convert input image to grayscale
        input_img_proc = rgb2gray(input_img_proc)
        # Resize image while maintaining aspect ration
        w,h,c = input_img.shape
        aspect_ratio = detect_size / w
        input_img_proc = resize(input_img_proc, (aspect_ratio*w, aspect_ratio*h)) * 255
        # Detect face bounding box
        bounding_box = self.face_detector(input_img_proc.astype(np.uint8), 1)
        cropped_face = None
        FACE_DETECT_FLAG = False
        # Crop face region if detected
        if len(bounding_box) != 0:
            # Convert mmod.rectangles to dlib.rectangle
            bounding_box = bounding_box[0].rect
            # Get bounding box local coordinates
            y0,x0 = bounding_box.tl_corner().x, bounding_box.tl_corner().y
            y1,x1 = bounding_box.tr_corner().x, bounding_box.bl_corner().y
            cx,cy = bounding_box.center().x, bounding_box.center().y
            coords = np.array([x0,x1,y0,y1,cx,cy])
            # Fit bounding box to original input image
            coords = coords / aspect_ratio
            coords = coords.astype(np.int)
            # Enlarge the bounding box
            coords[0],coords[2] = coords[0] / scale, coords[2] / scale
            coords[1],coords[3] = coords[1] * scale, coords[3] * scale
            coords = coords.astype(np.int)
            # Crop facial region
            cropped_face = input_img[coords[0]:coords[1],coords[2]:coords[3],:]

            # Save bounding box information
            self.bbox = coords
            # Update flag
            FACE_DETECT_FLAG = True

        return cropped_face, FACE_DETECT_FLAG

    def visualize_pcl(self, uv_map):
        '''
        * Note: still experimental
        
        INPUT : [1] uv_map; UV representation of the 3D face, numpy array
        OUTPUT: [1] poincloud; Dense 3D face
        '''
        # Infer pointcloud data from UV-Map
        pcl = np.reshape(uv_map.copy(), [self.resolution*self.resolution, -1])
        pcl_ori = pcl.copy()
        # Make empty frame for point cloud visualization
        frame_size = 500
        scale = 400
        tr = 50
        pcl_frame = np.zeros((frame_size, frame_size, 3))
        #--- Fit pointcloud to empty frame
        #--  Scale the point cloud  
        pcl = pcl * scale
        #--  Translate the point cloud to the center of frame
        pcl = pcl + 50
        #--  Convert data type to integer
        pcl = pcl.astype(np.int)
        #--  Create color map for depth (Z-coordinates)
        # color_map = LinearSegmentedColormap.from_list('PCL Color', [''])
        color_map = mpl.cm.get_cmap('jet', 12)
        depth_map = color_map(uv_map[:,:,-1])
        depth_map = (depth_map*255).astype(np.int)
        # Make point cloud frame
        pcl_frame = np.zeros((frame_size, frame_size, 3))
        pcl_frame[pcl[:,0], pcl[:,1], :] = np.reshape(depth_map[:,:,:3], [self.resolution*self.resolution, -1])
        pcl_frame = np.rot90((np.rot90(np.rot90(pcl_frame))))
        pcl_frame = np.fliplr(pcl_frame)


        return pcl_frame, pcl_ori