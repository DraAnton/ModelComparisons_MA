import cv2
import albumentations as A
import numpy as np

from typing import List, Dict, Callable, Tuple


# Function that corrects bounding boxes that may lie outside of the frame
def Clip(image: np.array, bboxes: np.array, labels: np.array) -> Tuple:
  width = image.shape[1]
  height = image.shape[0]

  out_bboxes = []
  for bb in bboxes:
    curr = [0,0,0,0]
    curr[0] = max(min(bb[0], width-2), 0)
    curr[1] = max(min(bb[1], height-2), 0)
    curr[2] = max(min(bb[2], width-1), curr[0] + 1)
    curr[3] = max(min(bb[3], height-1), curr[1] + 1)
    out_bboxes.append(curr)
  return image, np.array(out_bboxes), labels

# Function that divides all pixes values by 255
def Normalize(image: np.array, bboxes: np.array, labels: np.array ) -> Tuple:
   return image / 255, bboxes, labels


# Class that can be initialized with several augmentation functions. In Each epoch, one random function is choosen from the function selection
#    ie. Random_Augmentation_Chooser([fun1, fun2, fun3], probabilities = [.8, .1, .1]) ----- in that case fun1 is more likely to be applied to the image
class Random_Augmentation_Chooser():
  def __init__(self, enhancements: List[Callable], probabilities: List[float] = None):
    self.own_functions = enhancements
    if(probabilities is None):
      self.choice_list = list(range(0,len(enhancements)))
    else:
      assert len(enhancements) == len(probabilities)
      assert sum(probabilities) > 0.95 and sum(probabilities) < 1.05
      self.choice_list = []
      for index, elem in enumerate(probabilities):
          self.choice_list = self.choice_list + [index]*int(100*elem)
      
  def __call__(self, image: np.ndarray, bboxes = None, labels = None):
    function_chooser = random.sample(self.choice_list, 1)[0]
    return self.own_functions[function_chooser](image, bboxes = bboxes, labels = labels)

# Resizes image and corrects the bounding boxes accordingly
class Resize():
  def __init__(self, width: int, height:int):
    self.width = width
    self.height = height
  
  def __call__(self, image: np.array, bboxes: np.array, labels: np.array) -> Tuple:
      new_boxes = []
      xShape = image.shape
      for bb in bboxes:
        x1 = bb[0]/xShape[1]*self.width
        y1 = bb[1]/xShape[0]*self.height
        x2 = bb[2]/xShape[1]*self.width
        y2 = bb[3]/xShape[0]*self.height
        new_boxes.append([x1, y1, x2, y2])

      new_boxes = np.array(new_boxes)
      image = cv2.resize(image, (self.width, self.height))
      return image, new_boxes, labels 


# Runs augmentations. First the custom augmentations followed by the augmentations from the Albumenation library
class Augmenter():
    def __init__(self, own_function: List[Callable], A_transform:A.core.composition.Compose) -> None:
        self.own_functions = own_function
        self.A_transform = A_transform
  
    def __call__(self, image: np.array, bboxes: np.array, labels: np.array):
        for f in self.own_functions:
            image, bboxes, labels = f(image = image, bboxes = bboxes, labels = labels)
        sample = self.A_transform(image = image, bboxes = bboxes, labels = labels)
        return sample["image"], sample["bboxes"], sample["labels"]

