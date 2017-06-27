"""
Contains common functionality that is used by different classes
(e.g. saving images, writing to files)
The idea of this class is to reduce importing the same packages over and over in different classes.
Methods will be added to this class as more code gets reformatted
"""

# imports
import os.path

import csv
import mahotas
import cv2

import scripts.preprocessing.preprocessing as preprocessing
import scripts.segmentation.segmentation as segmentation
from . import main

try:
    import tkinter as tk  # for python 3
except:
    import Tkinter as tk  # for python 2

# take care of the path for different OS
os_path = str(os.path)
if 'posix' in os_path:
    import posixpath as path
elif 'nt' in os_path:
    import ntpath as path


"""
Save/read/write methods
"""


def save_image(pathpart1, pathpart2, imagefile):
    """
    Method for GUI display, save an image somewhere
    :param path: where to save the image
    :param imagefile: the actual image
    """

    joined_path = join_path(pathpart1, pathpart2)
    mahotas.imsave(joined_path, imagefile)


def read_image(pathpart1, pathpart2):
    """
    OpenCV method, read image from path
    """
    return cv2.imread(join_path(pathpart1, pathpart2))


def write_image(pathpart1, pathpart2, imagefile):
    """
    OpenCV method, write image to path
    """
    joined_path = join_path(pathpart1, pathpart2)
    cv2.imwrite(joined_path, imagefile)


def csv_writer(openedfile):
    """
    Return a csv writer for a file
    """
    return csv.writer(openedfile, lineterminator='\n')


"""
Methods for image and GUI manipulation
"""


def resize_image(imagefile, dimension):
    """
    OpenCV method, return resized image.
    :param dimension: requested dimension of the new image
    """
    return cv2.resize(imagefile, dimension, interpolation=cv2.INTER_AREA)


def tkinter_photoimage(filepath):
    """
    Tkinter method, return image
    """
    return tk.Photoimage(file=filepath)


def display_image(image):
    """Use the application root to display an image
    """
    root = main.get_root()
    root.displayimage = image


def draw_str(dst, target, s):
    """ Puts text onto an image
    :param dst: the image on which to put a string
    :param target: a tuple designating the image coordinates of the desired text
    :param s: the string to draw on the image
    """
    x, y = target
    cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN,
                1.0, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN,
                1.0, (255, 255, 255), lineType=cv2.LINE_AA)


def displaycoordinates(self, objectLabel, a, b, tm):
    """ Display coordinates of the cell to the panel"""

    if objectLabel == 0:
        self.label_10.configure(text=int(tm))
        self.label_43.configure(text=a)
        self.label_63.configure(text=b)
    if objectLabel == 1:
        self.label_11.configure(text=int(tm))
        self.label_44.configure(text=a)
        self.label_64.configure(text=b)

    if objectLabel == 2:
        self.label_12.configure(text=int(tm))
        self.label_45.configure(text=a)
        self.label_65.configure(text=b)

    if objectLabel == 3:
        self.label_13.configure(text=int(tm))
        self.label_46.configure(text=a)
        self.label_66.configure(text=b)

    if objectLabel == 4:
        self.label_14.configure(text=int(tm))
        self.label_47.configure(text=a)
        self.label_67.configure(text=b)

    if objectLabel == 5:
        self.label_15.configure(text=int(tm))
        self.label_48.configure(text=a)
        self.label_68.configure(text=b)
    if objectLabel == 6:
        self.label_16.configure(text=int(tm))
        self.label_49.configure(text=a)
        self.label_69.configure(text=b)
    if objectLabel == 7:
        self.label_17.configure(text=int(tm))
        self.label_50.configure(text=a)
        self.label_70.configure(text=b)

    if objectLabel == 8:
        self.label_18.configure(text=int(tm))
        self.label_51.configure(text=a)
        self.label_71.configure(text=b)

    if objectLabel == 9:
        self.label_19.configure(text=int(tm))
        self.label_52.configure(text=a)
        self.label_72.configure(text=b)


"""
Methods for forwarding external modules
"""


def join_path(pathpart1, pathpart2):
    """
    Join the path parts using the intelligent joiner from the imported module
    """

    return path.join(pathpart1, pathpart2)


"""
Other methods
"""


def concatenate_list(arg):
    """Concatenate lists
    """
    if isinstance(arg, (list, tuple)):
        for element in arg:
            for elem in concatenate_list(element):
                yield elem
    else:
        yield arg


def call_preprocessing(image, preprocessing_method):
    """
    Execute a preprocessing method on an image
    """

    if preprocessing_method == 1:
        preprocessed_image = preprocessing.histEqualize(image)

    elif preprocessing_method == 2:
        preprocessed_image = preprocessing.brightening(image)

    elif preprocessing_method == 3:
        preprocessed_image = preprocessing.GaussianBlurring(image)

    elif preprocessing_method == 4:
        preprocessed_image = preprocessing.darkening(image)

    elif preprocessing_method == 5:
        preprocessed_image = preprocessing.denoising(image)

    elif preprocessing_method == 6:
        preprocessed_image = preprocessing.binaryThresholding(image)

    elif preprocessing_method == 8:
        preprocessed_image = preprocessing.sharpening(image)

    if preprocessing_method == 7:

        preprocessed_image = image

    return preprocessed_image


def call_segmentation(segmentationmethod, preimage, rawimage, min_areasize, max_areasize, fixscale, min_distance, cell_estimate, color, thre):
    """ Call segmentation methods
    :param segmentationmethod: segmentation methods
    :param preimage: input image to segment
    :param rawimage: raw image without preprocessing
    :param min_areasize: estimated minimum area size of the cell
    :param max_areasize: estimated maximum area size of the cell
    :param fixscale: pixel intensity from 0.1-1.0
    :param min_distance: the minimum distance between the cells
    :param cell_estimate: minimum estimated number of cells per image
    :param color: color cell path """

    initialpoints, boxes, mask_image, image, cellmorph, processedimage = [], [], [], [], [], []
    processedimage = preimage

    if segmentationmethod == 1:
        initialpoints, boxes, mask_image, mage = segmentation.blob_seg(
            processedimage)

    if segmentationmethod == 2:
        if color == 1:
            initialpoints, boxes, mask_image, image, cellmorph = segmentation.black_background(
                processedimage, rawimage, min_areasize, max_areasize)

        if color == 2:
            initialpoints, boxes, mask_image, image, cellmorph = segmentation.white_background(
                processedimage, rawimage, min_areasize, max_areasize)

    if segmentationmethod == 3:
        initialpoints, boxes, image = segmentation.harris_corner(
            processedimage, int(cell_estimate), float(fixscale), int(min_distance))

    if segmentationmethod == 4:
        initialpoints, boxes, image = segmentation.shi_tomasi(
            processedimage, int(cell_estimate), float(fixscale), int(min_distance))

    if segmentationmethod == 5:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.kmeansSegment(
            processedimage, rawimage, 1, min_areasize, max_areasize)

    if segmentationmethod == 6:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.graphSegmentation(
            processedimage, rawimage, min_areasize, min_areasize, max_areasize)

    if segmentationmethod == 7:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.meanshif(
            processedimage, rawimage, min_areasize, max_areasize, int(fixscale * 100))

    if segmentationmethod == 8:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.sheetSegment(
            processedimage, rawimage, min_areasize, max_areasize)

    if segmentationmethod == 9:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.findContour(
            processedimage, rawimage, min_areasize, max_areasize)

    if segmentationmethod == 10:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.threshold(
            processedimage, rawimage, min_areasize, max_areasize)

    if segmentationmethod == 11:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.overlapped_seg(
            processedimage, rawimage, min_areasize, max_areasize)

    if segmentationmethod == 12:
        initialpoints, boxes, mask_image, image, cellmorph = segmentation.gredientSeg(processedimage, rawimage,
                                                                                      min_areasize, max_areasize, thre)

    return initialpoints, boxes, mask_image, image, cellmorph