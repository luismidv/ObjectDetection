import pandas as pd
import pathlib
import os
from bs4 import BeautifulSoup
from PIL import Image
from tzdata import IANA_VERSION
import numpy as np
from torch.utils.data import Dataset
import supervision as sv
import cv2
import splitfolders

class Dataset(Dataset):
    def __init__(self, images, bbox_data, image_size, name,train_val_route, test_route):
        super().__init__()
        print("Initializing Dataset")
        self.images = images
        self.bbox_data = bbox_data
        self.image_size = image_size
        self.name  = name
        self.train_val_route = train_val_route
        self.test_route = test_route


    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        print("Hey")

def prepare_diferent_data(data_path):
    # TRAINING DATA
    image_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
    anotations_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations')
    imagesets_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets')
    segmentation_object = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/SegmentationObject')
    segmentation_directory = data_path.joinpath('VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/SegmentationClass')

    # TESTING DATA
    image_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
    anotations_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations')
    imagesets_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets')
    anotations_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/SegmentationObject')
    segmentation_directory_test = data_path.joinpath('VOCtest_06-Nov-2007/VOCdevkit/VOC2007/SegmentationClass')

    return image_directory,anotations_directory


def extract_info_xml(annotations_set):
    document_dict = {}
    file_list = os.listdir(annotations_set)
    for file in file_list:
        document_dict[file] = []
        new_path = os.path.join(annotations_set, file)
        try:
            with open(new_path,'r') as f:
                data = f.read()
                bs_data = BeautifulSoup(data, 'lxml-xml')
                bs_filename = bs_data.find('filename').text
                bs_objects = bs_data.find_all('object')

                obj_number = 0
                for obj in bs_objects:

                    obj_number+=1
                    bs_x_min = obj.find('xmin').text
                    bs_y_min = obj.find('ymin').text
                    bs_x_max = obj.find('xmax').text
                    bs_y_max = obj.find('ymax').text
                    object_name = obj.find('name').text
                    new_list =  [len(bs_objects), obj_number, bs_x_min,bs_y_min, bs_x_max, bs_y_max,object_name]
                    document_dict[file].append(new_list)

        except Exception as error:
            print(f"Error while reading xml file check it's sintax: {error}")
    return document_dict


def get_images_folder(data_dict):
    images_names,values = zip(*data_dict.items())
    images_names = list(images_names)
    values = list(values)


    position = 0
    for image in list(images_names):
        image = image.replace('.xml', '.jpg')
        images_names[position] = image
        position+=1
    data_dict = dict(zip(images_names, values))


    return data_dict

def get_image_with_data(data_dict, imageroute):
    image_list = []

    image_directory = pathlib.Path(imageroute)
    image_names = data_dict.keys()
    image_names = list(image_names)
    for image in image_names:
        image_path = image_directory.joinpath(image)

        try:
            new_image = Image.open(image_path)
            new_image = np.array(new_image)
            image_list.append(new_image)

        except Exception as error:
            print(f"Image {image} failed at loading")

    return image_list
def image_show_fn(document_dict,image_folder):
    images,bbox = zip(*document_dict.items())
    images = list(images)
    bbox = list(bbox)
    bbox_list = bbox[0]
    #If you want to initialize the array length to the total of objects in the image
    #total_objects = bbox_list[0][0]
    anotations_array = np.empty((0,4))
    every_object = []
    for bbox in bbox_list:
        print(f"Bbox for the image on: {images[0]}")

        x1 = int(bbox[2])
        y1 = int(bbox[3])
        x2 = int(bbox[4])
        y2 = int(bbox[5])

        object_name = bbox[-1]
        every_object.append(object_name)

        total_objects = int(bbox[0])
        coord_array = np.array([[x1,y1,x2,y2]])

        print(f"Bounding box coordinate x1 {x1} coordinate x2 {x2} coordinate y1 {y1} coordinate y2 {y2}")
        image = os.path.join(image_folder, images[0])
        image = cv2.imread(image)

        anotations_array = np.append(anotations_array,coord_array,axis=0)
        array_confidence = np.random.uniform(0.50,0.95, total_objects)

    print(f"Every object name {every_object}")
    array_ids = np.array([i for i in range(0,total_objects)])
    detections = sv.Detections(
        xyxy = anotations_array,
        class_id=array_ids,
        confidence=array_confidence
    )

    bounding_box_anotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    anotated_frame = bounding_box_anotator.annotate(
        scene = image.copy(),
        detections = detections,

    )
    anotated_frame = label_annotator.annotate(
        scene = anotated_frame,
        detections = detections,
        labels = every_object
    )
    sv.plot_image(anotated_frame)


def bbox_debugging(data_dict):
    images, bbox = zip(*data_dict.items())
    print(bbox)

data_directory = pathlib.Path("./cfg/archive")
image_directory, anotations_directory = prepare_diferent_data(data_directory)
#data_treat_check(image_directory, anotations_directory)

document_dict = extract_info_xml(anotations_directory)

data_dict = get_images_folder(document_dict)
image_list = get_image_with_data(data_dict, './cfg/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages')
image_show_fn(data_dict,'./cfg/archive/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages' )
#bbox_debugging(data_dict)