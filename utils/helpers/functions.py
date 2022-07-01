import cv2, os
import numpy as np

from typing import List, Tuple


# draws bounding box onto image and displays it with cv2.imshow()
def boxes_show(img: np.array, boxes: List[Tuple]) -> None:
  for elem_dry in boxes:
    elem = list(int(e) for e in elem_dry)
    img = cv2.rectangle(img, (elem[0], elem[1]), (elem[2], elem[3]), (0,0,255), 2)
  cv2.imshow(img)

# draws bounding box onto image and returns the resulting array
def boxes_draw(img: np.array, boxes: List[Tuple], color: Tuple = (255,0,0), size: int = 2) -> np.array:
  for elem_dry in boxes:
    elem = list(int(e) for e in elem_dry)
    img = cv2.rectangle(img, (elem[0], elem[1]), (elem[2], elem[3]), color, size)
  if( type(img) == np.ndarray ):
    return img
  return img.get()


def image_and_label_paths(path: str, image_dir: str = "images", label_dir: str = "labels", valid_file_types: List[str] = ["csv", "xml"] ) -> Tuple:
  path = path if path[-1] == "/" else path+"/"
  path_images = path+image_dir+"/"
  path_boxes = path+label_dir+"/"

  input_images = [elem for elem in os.listdir(path_images)]
  input_boxes = [elem for elem in os.listdir(path_boxes) if elem.split(".")[-1] in valid_file_types]
  no_file_endings = [elem.split(".")[0] for elem in input_boxes]

  counter = 0
  len_inputs_images = len(input_images)
  while(counter < len_inputs_images):
    current = input_images[counter].split(".")[0]
    if(current not in no_file_endings):
      input_images = input_images[:counter] + input_images[counter+1:]
      len_inputs_images = len_inputs_images - 1
    else:
      counter += 1 

  if(len(input_images) != len(input_boxes)):
    no_file_endings = [elem.split(".")[0] for elem in input_images]
    
    counter = 0
    len_inputs_boxes = len(input_boxes)
    while(counter < len_inputs_boxes):
      current = input_boxes[counter].split(".")[0]
      if(current not in no_file_endings):
        input_boxes = input_boxes[:counter] + input_boxes[counter+1:]
        len_inputs_boxes = len_inputs_boxes - 1
      else:
        counter += 1 
  
  input_images.sort()
  input_boxes.sort()
  return ([path_images + elem for elem in input_images], [path_boxes + elem for elem in input_boxes])
