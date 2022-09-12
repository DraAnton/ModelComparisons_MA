from utils import DataSet
import utils.augmentation as aug
import utils.helpers as helpers

import torch, torchvision 
import albumentations as A
import cv2
import numpy as np 
import pandas as pd
import os, random, time, json, math
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import train_test_split
#from mean_average_precision import MetricBuilder
from tqdm.auto import tqdm

#from ImageEnhancement import MSRCR, FUSION, CLAHE


BATCH_SIZE = 2 # increase / decrease according to GPU memeory
RESIZE_TO = 800 # resize the image for training and transforms
NUM_EPOCHS = 5 # number of epochs to train for
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
SEED = 42
TEST_RATIO = 0.1 # for train/test split

MAPPING = {
    'DUMMY': 0,
    'Fish': 1,
    'Cnidaria':2   
}

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False
# location to save model and plots
#OUT_DIR = '/home/anton/Documents/Thesis/temResults'
#OUT_DIR = "/content/res"
OUT_DIR = "./"
SAVE_PLOTS_EPOCH = 5 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 5 # save model after these many epochs

PREPARE_TEST_DATA = True

IMAGE_DIRECTORY = "images"








# Getting all relevant paths for images and their respective label files and splitting them 
#      into train and validation datasets

base_dir = "/data"
#base_dir = "/home/anton/Downloads/image_annotator/linux_v1.4.3/data/oldclean/"
#base_dir = "/content/drive/MyDrive/ROV_ECIM/multimedia/ECIM_bruv_data"
imgs, labels = helpers.image_and_label_paths(base_dir, image_dir = "images", label_dir = "labels")
inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(imgs[:9],labels[:9], test_size=TEST_RATIO, random_state=SEED)


# Train DataSet:
# Augmentations from the Albumentations library
albumentations_augmentations = A.Compose([ A.Flip(0.45), A.RandomRotate90(0.5),
                                           A.MotionBlur(p=0.2), A.Blur(blur_limit=3, p=0.1)],
                                         bbox_params={  'format': 'pascal_voc', 
                                                        'label_fields': ['labels']}
                                        )

# custum augmentations combined with those from the library
my_albumenations = aug.Augmenter([aug.Clip, aug.Resize(800, 800), aug.Normalize], 
                                                 albumentations_augmentations)

# create DataSet with correct sample lists
train_dataset = DataSet(inputs_train, 
                            targets_train, 
                            use_cache          = False,
                            transform          = my_albumenations,
                            mapping            = MAPPING,
                            random_enhancement = False
                            )



# Validation DataSet:
# Augmentations from the Albumentations library
albumentations_augmentations = A.Compose([],bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# custum augmentations combined with those from the library
my_albumenations = aug.Augmenter([aug.Clip, aug.Resize(800, 800), aug.Normalize], 
                                                 albumentations_augmentations)


# create DataSet object with correct sample lists
validation_dataset = DataSet(inputs_valid, 
                                targets_valid, 
                                use_cache          = False,
                                transform          = my_albumenations,
                                mapping            = MAPPING, 
                                random_enhancement = False)          

# Example for accessing entries in a DataSet object:
#train_dataset[0]




def collate_fn(batch):
    return tuple(zip(*batch))

# A Torch dataloader takes our dataset object to successively extract same sized batches from it 
train_loader = DataLoader(  dataset     = train_dataset,
                            batch_size  = BATCH_SIZE,
                            shuffle     = True,
                            num_workers = 0,
                            collate_fn  = collate_fn
                         )


valid_loader = DataLoader(  dataset     = validation_dataset,
                            batch_size  = BATCH_SIZE,
                            shuffle     = False,
                            num_workers = 0,
                            collate_fn  = collate_fn
                         )

#train_loss_hist = Averager()
#val_loss_hist = Averager()


# function that creates model objects and download pretrained weights
def create_model(num_classes):
    #input_pths = [elem for elem in os.listdir("./") if elem.split(".")[-1] == "pth"]
    #if(len(input_pths) == 0):
    #    raise ValueError("pth file for pretrained model not found")
    #print("HALLO")
    #model_dict = torch.load(input_pths[0])
    #print(model_dict.keys())
    #model = FastRCNNPredictor(1024, num_classes) 
    #model.load_state_dict(model_dict)

    
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    #model = FastRCNNPredictor(1024, num_classes) 
    #input_images = [elem for elem in os.listdir("./") if elem.split(".")[-1] == "pth"]
    #if(len(input_images) == 0):
    #    raise ValueError("pth file for pretrained model not found")
    #print(input_images[0])
    #model.load_state_dict(torch.load("./"+input_images[0]))

    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    #print(in_features)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

# function that executes training for one epoch
def train(train_data_loader, model, optimizer, train_itr, train_loss_list):
    print('Training')
    loss_avg = 0
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items() if k in ["boxes", "labels"]} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        loss_avg += loss_value
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list, train_itr, (1.0*loss_avg)/len(train_data_loader)

# function that executes validation for one epoch
def validate(valid_data_loader, model, val_itr, val_loss_list):
    print('Validating')
    loss_avg = 0
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items() if k in ["boxes", "labels"]} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        loss_avg += loss_value
        val_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list, val_itr, (1.0*loss_avg)/len(valid_data_loader)




MODEL_NAME = 'model'
MODEL_APPENDIX = "RANDOM_FINAL"

model = create_model(num_classes = len(MAPPING.keys()) )

### FOR IMPORTING A MODEL
#model.load_state_dict(torch.load(OUT_DIR+"/FILE-NAME.pth"))
###

model = model.to(DEVICE)

# define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.003, momentum=0.9, weight_decay=0.0001)

train_itr = 1
val_itr = 1
train_loss_list = []
val_loss_list = []


# start the training
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    # start timer and carry out training and validation
    start = time.time()
    train_loss_list, train_itr, train_loss_avg = train(train_loader, model, optimizer, train_itr, train_loss_list)
    val_loss_list, val_itr, val_loss_avg = validate(valid_loader, model, val_itr, val_loss_list)
    print(f"Epoch #{epoch} train loss: {train_loss_avg:.3f}")   
    print(f"Epoch #{epoch} validation loss: {val_loss_avg:.3f}")   
    end = time.time()
    
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    if (epoch+1) % SAVE_MODEL_EPOCH == 0 or (epoch+1) == NUM_EPOCHS: # save model after every n epochs or at end
        torch.save(model.state_dict(), f"{OUT_DIR}/model_{MODEL_APPENDIX}_{epoch+1}.pth")
        print('SAVING MODEL COMPLETE...\n')
    if (epoch+1) % SAVE_PLOTS_EPOCH == 0 or (epoch+1) == NUM_EPOCHS: # save loss plots after n epochs or at end
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        train_ax.plot(train_loss_list, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        valid_ax.plot(train_loss_list, color='red')
        valid_ax.set_xlabel('iterations')
        valid_ax.set_ylabel('validation loss')
        figure_1.savefig(f"{OUT_DIR}/train_loss_{MODEL_APPENDIX}_{epoch+1}.png")
        figure_2.savefig(f"{OUT_DIR}/valid_loss_{MODEL_APPENDIX}_{epoch+1}.png")
        print('SAVING PLOTS COMPLETE...') 