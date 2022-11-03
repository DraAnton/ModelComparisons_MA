import torch, cv2, re, random
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from typing import List, Dict, Tuple, Callable
from tqdm.auto import tqdm



def read_from_csv(csv_path: str, mapping:Dict, columnName:str ) -> Dict:
  df = pd.read_csv(csv_path)
  labels = []
  boxes = []
  valid_classes = mapping.keys()
  for index, row in df.iterrows():
    #labels.append("Fish")
    spc_ = row[columnName]
    assert spc_ in valid_classes, "Class {} not specified in mapping".format(spc_)
    labels.append(mapping[spc_])
    x_ = row["RectX"]
    y_ = row["RectY"]
    width_ = row["RectWidth"]
    height_ = row["RectHeight"]
    boxes.append([x_, y_, x_+width_, y_+height_])

  return {"labels":np.array(labels), "boxes":np.array(boxes)}


def read_from_xml(path: str, mapping: Dict) -> Dict:
  tree = ET.parse(path)
  root = tree.getroot()
  enddict = {"labels":[], "boxes":[]}
  valid_classes = mapping.keys()

  for child in root:
    if(child.tag == "object"):
      for infos in child:
        if(infos.tag == "name"):
          spc_ = infos.text
          assert spc_ in valid_classes, "Class {} not specified in mapping".format(spc_)
          enddict["labels"].append(mapping[spc_])
        elif(infos.tag == "bndbox"):
          curr_box = [0,0,0,0]
          for pos in infos:      
            if(pos.tag == "xmin"):
              curr_box[0] = int(pos.text)
              #curr_box[0] =  max(int(xsize - int(pos.text)), 1)
            elif(pos.tag == "ymin"):
              #curr_box[1] =  max(int(xsize - int(pos.text)), 1)    
              curr_box[1] = int(pos.text)
            elif(pos.tag == "xmax"):
              #curr_box[2] = max(int(xsize - int(pos.text)), 1)
              curr_box[2] = int(pos.text)
            elif(pos.tag == "ymax"):
              curr_box[3] = int(pos.text)
              #curr_box[3] = max(int(xsize - int(pos.text)), 1)
          enddict["boxes"].append(curr_box)
          #print(curr_box)
  a = np.array(enddict["labels"])
  b = np.array(enddict["boxes"])
  #print({"labels":a, "boxes":b})
  return {"labels":a, "boxes":b}


class READER_img_and_label(): # can read csv and xml files as labels. The right function is chosen here
    def __init__(self, mapping: Dict, csv_label_column: str):
        self.mapping = mapping
        self.csv_label_column = csv_label_column
    
    def __call__(self, inp: str, tar: str) -> Tuple:
        if inp is None:
            im_ = None
        else:
            im_ = cv2.imread(inp)
        tpe = tar.split(".")[-1]
        if(tpe == "csv"):
            a= read_from_csv(tar, self.mapping, self.csv_label_column)
            return im_, a
        return im_, read_from_xml(tar, self.mapping)


  
'''
Class: DataSet
  Function: __init__()
    inputs: List of strings representing the paths to the images
    targets: List of stings representing the paths to the BoundingBox Files (needs to be the same order as "inputs" List)
    use_cache: all images are read into memory -> no read operation has to be conducted during training (only for smaller training sets)
    convert_to_format: not used anymore
    mapping: dictionary that maps class present in the Label Files to integer (e.g. {"Fish":1, "Potato":2, "Human":3})
    random_enhancement: only for the specific use case of randomly enhanced images. When reading images, it will randomly choose from different version of the image present on the hard drive
    use_detectron: When the Mask-RCNN implementation is used. Different output format 
    csv_label_column: the csv files need to have a specific format. columns: (RectX, RectY, RectWidth, RectHeight, class). The column name "class" can be different and specified here
'''
class DataSet(torch.utils.data.Dataset):
    def __init__(self,
                inputs: List[str],
                targets: List[str],
                use_cache: bool = False,
                convert_to_format: str = None,
                transform: Callable = None,
                mapping: Dict = None,
                random_enhancement: bool = False,
                use_detectron: bool = False,
                csv_label_column: str = "class"
                ):
        assert len(inputs) == len(targets), "List of images and List of label files are not equally sized" 
        self.inputs = inputs
        self.targets = targets
        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping
        self.transform = transform
        self.random_enhancement = random_enhancement
        self.use_detectron = use_detectron
        self.image_and_label_reader = READER_img_and_label(self.mapping, csv_label_column)
        if(self.use_detectron):
            return 

        # without detectron it is possible to read all images beforehand and store them into ram(if sufficient memory is available)
        if self.use_cache and not random_enhancement:
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.image_and_label_reader, zip(inputs, targets))

    def __len__(self):
        return len(self.inputs)


    # USED when NOT using DETECTRONs buildin dataloader
    #     used like: datasetElemet[5] -> returns the fifth image and the respective bounding box coordinates
    def __getitem__(self, index: int):
      # Select the sample
      if self.use_cache: # IF ALREADY LOADED INTO MEMORY
        x, y = self.cached_data[index]
        x_name = self.inputs[index].split("/")[-1]
        y_name = self.targets[index].split("/")[-1]
        input_ID = self.inputs[index]
      else: # IF NOT LOADED INTO MEMORY -> image still needs to be read
        input_ID = self.inputs[index]
        target_ID = self.targets[index]
        if self.random_enhancement: # if different versions of the image exists in different directories
          enh = random.choice(["images", "msrcr", "clahe", "fusion"])
          input_ID = input_ID.replace("images", enh)
        
        x, y = self.image_and_label_reader(input_ID, target_ID)
        x_name = self.inputs[index].split("/")[-1]
        y_name = self.targets[index].split("/")[-1]

      # From RGBA to RGB
      if x.shape[-1] == 4:
        from skimage.color import rgba2rgb
        x = rgba2rgb(x)

      if self.transform is not None: # applies a image transformation pipeline
        x, boxes, labels = self.transform(image = x, bboxes = y["boxes"], labels = y["labels"])  # returns np.ndarrays
        y["boxes"] = np.array(boxes)
        y["labels"] = np.array(labels)

      # Typecasting
      x = np.moveaxis(x, source = -1, destination = 0)
      x = torch.from_numpy(x).type(torch.float32)
      y = {key: torch.from_numpy(value) for key, value in y.items()}
      y["image_id"] = input_ID
      return x, y

    # USED when using DETECTRONs buildin dataloader (reads images only to extract their dimensions)
    def get_data_dicts(self, name = "train"):
        assert self.use_detectron, "use_detectron flag needs to be set true"
        from detectron2.structures import BoxMode
        dataset_dicts = []

        idx = 0
        prog_bar = tqdm(zip(self.inputs, self.targets), total=len(self.inputs))
        print(f"loading {name} dataset")
        for im, lab in prog_bar:
            record = {}
        
            height, width = cv2.imread(im).shape[:2]
        
            record["file_name"] = im
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
      
            _, annos = self.image_and_label_reader(None, lab)
            objs = []
            for m_cls, box in zip(annos["labels"], annos["boxes"]):
                px = np.array([box[0], box[2]])
                py = np.array([box[1], box[3]])
                poly = [int(box[0] + int((box[2]-box[0])/2)), int(box[1]), int(box[0]), int(box[3]), int(box[2]), int(box[3])]

                obj = {
                    "bbox": [int(np.min(px)), int(np.min(py)), int(np.max(px)), int(np.max(py))],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": m_cls,
                }
                objs.append(obj)
            record["annotations"] = objs
            idx += 1
            dataset_dicts.append(record)

        return dataset_dicts #The return Dictionary does not contain the images, but only information about the images(e.g. the path for each image and boundingBox coordinates). Detectron handles the image reading internally