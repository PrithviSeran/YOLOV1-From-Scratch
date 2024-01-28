import pandas as pd
from PIL import Image
import numpy as np
import glob
import tensorflow as tf
import random
import cv2
from matplotlib import pyplot as plt
from tensorflow import keras
import keras.backend as K

# TAKEN FROM VIVEK MASKARA'S ARTICLE 'Implementing YOLOV1 from scratch using Keras Tensorflow 2.0'
class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B
    
    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs


# changing the notation from x_min, y_min, x_max, y_max to x_center, y_center, width, height
def xmin_to_xcenter(csv_file_path):
    data = pd.read_csv(csv_file_path)

    x_min = data['xmin']
    y_min = data['ymin']
    x_max = data['xmax']
    y_max = data['ymax']

    #print((((x_min + x_max)/2)/676))
    x_center = (((x_min + x_max)/2)/676)
    y_center = (((y_min + y_max)/2)/380)
    width = ((x_max - x_min)/676)
    height = ((y_max - y_min)/380)

    data['xmin'] = x_center
    data['ymin'] = y_center
    data['xmax'] = width
    data['ymax'] = height

    new = data.rename(columns={"xmin": "x_center", "ymin": "y_center", "xmax": "width", "ymax": "height"})

    new.to_csv(csv_file_path, index = False)

    print("Done! ")

def xcenter_to_xmin(csv_file_path):

    data = pd.read_csv(csv_file_path)

    x_center = data['x_center']
    y_center = data['y_center']
    width = data['width']
    height = data['height']

    xmin = ((x_center*676) - ((width*676)/2)) 
    ymin = ((y_center*380) - ((height*380)/2)) 
    xmax= ((x_center*676) + ((width*676)/2)) 
    ymax = ((y_center*380) + ((height*380)/2)) 

    data['x_center'] = xmin
    data['y_center'] = ymin
    data['width'] = xmax
    data['height'] = ymax

    new = data.rename(columns={"x_center": "xmin", "y_center": "ymin", "width": "xmax", "height": "ymax"})

    new.to_csv(csv_file_path, index = False)

    print("Done! ")

def xcenter_to_xmin_individual(x_center, y_center, width, height):

    xmin = int((x_center*448) - ((width*448)/2))
    ymin = int((y_center*448) - ((height*448)/2))
    xmax= int((x_center*448) + ((width*448)/2))
    ymax = int((y_center*448) + ((height*448)/2))

    return [xmin, ymin, xmax, ymax]

#resizing the images and saving the array to a file
def scale_resize_image(training_data, outfile):
    images_resized = []

    image_list = []
    
    for image in list(training_data['image']):
        im=np.asarray(Image.open("data/training_images/"+image))
        image_list.append(im)

    for image in image_list:
        new_image = tf.image.convert_image_dtype(image, tf.float32) # equivalent to dividing image pixels by 255
        new_image = tf.image.resize(new_image, (448, 448)) # Resizing the image to 224x224 dimention
        images_resized.append(new_image)

    np.save(outfile, images_resized)

    print("Done!")

# formate of output vectors: is object present, x, y, h, w, confidence
def getting_ground_truth_index(bounding_box):
    x_center = bounding_box[0]
    y_center = bounding_box[1]

    #print(bounding_box)
    #print(x_center * 448, y_center * 448)

    x_box = (int(x_center * 448) // 64) 
    y_box = (int(y_center * 448) // 64)

    #print(x_center)

    return (x_box, y_box)

def make_truth_matrixes(boxes):

    truth_matrixes = []

    for k in range(559):
        #print(k)
        truth_matrixes.append(make_truth_matrix(boxes[k]))

    return np.array(truth_matrixes)

    """
    truth_matrixes = np.zeros((training_image_len, 7, 7, 6))

    for index in range(training_image_len):

        #print(images.index(training_set_data['image'][index]))

        truth_matrixes[index] = make_truth_matrix(training_set_data['x_center'][index], training_set_data['y_center'][index], training_set_data['width'][index], training_set_data['height'][index])

    print("Done!")

    return truth_matrixes
    """

def make_truth_matrix(box):

    label_matrix = np.zeros([7, 7, 30])

    cls = 1
    x = box[0] 
    y = box[1] 
    w = box[2] 
    h = box[3]
    loc = [7 * x, 7 * y]
    loc_i = int(loc[1])
    loc_j = int(loc[0])

    if label_matrix[loc_i, loc_j, 24] == 0:
        #print(y, x)
        label_matrix[loc_i, loc_j, cls] = 1
        label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
        label_matrix[loc_i, loc_j, 24] = 1  # response

    return label_matrix

# format of output vectors: is object present, x, y, h, w, confidence
def calculate_loss(y_truth, y_hat, lambda_cor = 5, lambda_noobj = 0.5):

    if_object_present = tf.cast(tf.expand_dims(y_truth[..., 0], -1), tf.float32)

    #print(if_object_present.shape)

    #part 1

    #print(y_truth[...,1:5].shape)
    #print("y_truth: ", type(y_truth[..., 0][0]))
    #print("y_hat: ", type(y_hat[..., 0]))

    box_targets = if_object_present * y_truth[:,:,:, 1:5]
    box_predictions = if_object_present * y_hat[:,:,:, 1:5]

    box_predictions = box_predictions.numpy()
    box_targets = box_targets.numpy()

    box_predictions[..., 2:4] = tf.sign(box_predictions[..., 2:4]) * tf.sqrt(tf.abs(box_predictions[..., 2:4] + 1e-6))
    box_targets[:,:,:, 2:4] = tf.sqrt(box_targets[:,:,:, 2:4])

    box_predictions = tf.convert_to_tensor(box_predictions)
    box_targets = tf.convert_to_tensor(box_targets)

    box_loss = tf.reduce_mean(tf.square(box_targets - box_predictions))

    # part 3 (object confidence loss)

    confidence_hat = if_object_present * y_hat[..., 5:6]
    confidence_truth = if_object_present * y_truth[..., 5:6]


    object_confidence_loss = tf.reduce_mean(tf.square(confidence_truth - confidence_hat))

    # part 4 (no object confidence loss)

    no_confidence_hat = (1 - if_object_present) * y_hat[..., 5:6]
    no_confidence_truth = (1 - if_object_present) * y_truth[..., 5:6]

    no_object_confidence_loss = tf.reduce_mean(tf.square(no_confidence_truth - no_confidence_hat))
    #print(box_loss)

    loss = lambda_cor * box_loss + object_confidence_loss + lambda_noobj * no_object_confidence_loss

    return loss

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

def iou(pred_mins, pred_maxes, true_mins, true_maxes):

    #print(K.maximum(tf.constant([3.0, 5.0]), tf.constant([5.0, 7.0])))
    #print(type(3.0))
    pred_mins = tf.cast(pred_mins, dtype='float32')
    true_mins = tf.cast(pred_mins, dtype='float32')
    pred_maxes = tf.cast(pred_mins, dtype='float32')
    true_maxes = tf.cast(pred_mins, dtype='float32')
    #print(type(pred_mins))

    #print(true_mins[0])

    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores

def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh

def yolo_loss(y_true, y_pred):

    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 24]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / 448)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss






