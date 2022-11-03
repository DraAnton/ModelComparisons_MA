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


def collate_fn(batch):
    return tuple(zip(*batch))

def prepare_data_loaders(hparams):
    # Getting all relevant paths for images and their respective label files and splitting them 
    #      into train and validation datasets

    base_dir = hparams["DATA_DIR"]

    imgs, labels = helpers.image_and_label_paths(base_dir, image_dir = hparams["IMAGE_DIRECTORY"], label_dir = "labels")
    inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(imgs[:9],labels[:9], test_size=hparams["TEST_RATIO"], random_state=hparams["SEED"])


    # Train DataSet:
    # Augmentations from the Albumentations library
    albumentations_augmentations = A.Compose([ A.Flip(0.45), A.RandomRotate90(0.5),
                                               A.MotionBlur(p=0.2), A.Blur(blur_limit=3, p=0.1)],
                                             bbox_params={  'format': 'pascal_voc', 
                                                            'label_fields': ['labels']}
                                            )

    # custum augmentations combined with those from the library
    resize_to = hparams["RESIZE_TO"]
    my_albumenations = aug.Augmenter([aug.Clip, aug.Resize(resize_to, resize_to), aug.Normalize], 
                                                     albumentations_augmentations)

    # create DataSet with correct sample lists
    train_dataset = DataSet(inputs_train, 
                                targets_train, 
                                use_cache          = False,
                                transform          = my_albumenations,
                                mapping            = hparams["MAPPING"],
                                random_enhancement = False
                                )



    # Validation DataSet:
    # Augmentations from the Albumentations library
    albumentations_augmentations = A.Compose([],bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


    # custum augmentations combined with those from the library
    my_albumenations = aug.Augmenter([aug.Clip, aug.Resize(resize_to, resize_to), aug.Normalize], 
                                                     albumentations_augmentations)


    # create DataSet object with correct sample lists
    validation_dataset = DataSet(inputs_valid, 
                                    targets_valid, 
                                    use_cache          = False,
                                    transform          = my_albumenations,
                                    mapping            = hparams["MAPPING"], 
                                    random_enhancement = False)          

    # Example for accessing entries in a DataSet object:
    #train_dataset[0]


    # A Torch dataloader takes our dataset object to successively extract same sized batches from it 
    train_loader = DataLoader(  dataset     = train_dataset,
                                batch_size  = hparams["BATCH_SIZE"],
                                shuffle     = True,
                                num_workers = 0,
                                collate_fn  = collate_fn
                             )


    valid_loader = DataLoader(  dataset     = validation_dataset,
                                batch_size  = hparams["BATCH_SIZE"],
                                shuffle     = False,
                                num_workers = 0,
                                collate_fn  = collate_fn
                             )
    return train_loader, valid_loader



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
def train(train_data_loader, model, optimizer, train_itr, train_loss_list, DEVICE):
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
def validate(valid_data_loader, model, val_itr, val_loss_list, DEVICE):
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



def start_training(train_loader, valid_loader, hparams):
    MODEL_NAME = 'model'
    MODEL_APPENDIX = "RANDOM_FINAL"

    MAPPING = hparams["MAPPING"]
    OUT_DIR = hparams["OUT_DIR"]
    DEVICE = hparams["DEVICE"]
    NUM_EPOCHS = hparams["NUM_EPOCHS"]
    SAVE_MODEL_EPOCH = hparams["SAVE_MODEL_EPOCH"]
    SAVE_PLOTS_EPOCH = hparams["SAVE_PLOTS_EPOCH"]

    model = create_model(num_classes = len(MAPPING.keys()) )

    ### FOR IMPORTING A MODEL
    #model.load_state_dict(torch.load(OUT_DIR+"/FILE-NAME.pth"))
    ###

    model = model.to(DEVICE)
    # define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=hparams["learning_rate"], momentum=hparams["momentum"], weight_decay=hparams["weight_decay"])

    train_itr = 1
    val_itr = 1
    train_loss_list = []
    val_loss_list = []
    val_loss_epochs = []


    # start the training
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # start timer and carry out training and validation
        start = time.time()
        train_loss_list, train_itr, train_loss_avg = train(train_loader, model, optimizer, train_itr, train_loss_list, DEVICE)
        print(f"Epoch #{epoch} train loss: {train_loss_avg:.3f}")   
        end = time.time()

        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        if (epoch+1) % SAVE_MODEL_EPOCH == 0 or (epoch+1) == NUM_EPOCHS: # save model after every n epochs or at end
            torch.save(model.state_dict(), f"{OUT_DIR}/model_{MODEL_APPENDIX}_{epoch+1}.pth")
            print('SAVING MODEL COMPLETE...\n')
        if (epoch+1) % SAVE_PLOTS_EPOCH == 0 or (epoch+1) == NUM_EPOCHS: # save loss plots after n epochs or at end
            val_loss_list, val_itr, val_loss_avg = validate(valid_loader, model, val_itr, val_loss_list, DEVICE)
            print(f"Epoch #{epoch} validation loss: {val_loss_avg:.3f}")   
            val_loss_epochs.append(epoch)
            
            figure_1, train_ax = plt.subplots()
            figure_2, valid_ax = plt.subplots()
            train_ax.plot(train_loss_list, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss_epochs, val_loss_list, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            figure_1.savefig(f"{OUT_DIR}/train_loss_{MODEL_APPENDIX}_{epoch+1}.png")
            figure_2.savefig(f"{OUT_DIR}/valid_loss_{MODEL_APPENDIX}_{epoch+1}.png")
            print('SAVING PLOTS COMPLETE...') 


class Hyperparameters():
    ERROR_DUPLICATE_PARAM = "Hyperparameter {} provided more than once"

    def __init__(self, *args):
        self.parameter_dict = {}
        for arg in args:
            assert type(arg) == tuple      
            assert parameter_dict.get(arg[0], None) is None, ERROR_DUPLICATE_PARAM.format(arg[0])
            self.parameter_dict[arg[0]] = arg[1]
    
    def __getitem__(self, item):
        return self.parameter_dict[item]

    def __setitem__(self, key, val):
        assert self.parameter_dict.get(key, None) is None, ERROR_DUPLICATE_PARAM.format(key)
        self.parameter_dict[key] = val

def listEntriesRecursive(base_path):
    out_dir = {"files":[]}
    for file in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path,file)):
            out_dir[file] = listEntriesRecursive(os.path.join(base_path,file))
        else:
            out_dir["files"].append(file)
    return out_dir

if __name__ == "__main__":
    # initialize training parameters
    hparams = Hyperparameters()
    hparams["OUT_DIR"] = os.environ["OUT_DIR"]
    hparams["HYPERPARAM_FILE"] = os.environ["HYPERPARAM_FILE"]
    hparams["DATA_CONF_FILE"] = os.environ["DATA_CONF_FILE"]
    hparams["DATA_DIR"] = os.environ["DATA_DIR"]

    with open(hparams["HYPERPARAM_FILE"]) as f:
        hparam_file_dict = json.loads(f.read())

    hparams["BATCH_SIZE"]       =   int(hparam_file_dict.get("BATCH_SIZE", 8))
    hparams["NUM_EPOCHS"]       =   int(hparam_file_dict.get("NUM_EPOCHS", 50))
    hparams["TEST_RATIO"]       =   float(hparam_file_dict.get("TEST_RATIO", 0.1))
    hparams["learning_rate"]    =   float(hparam_file_dict.get("learning_rate", 0.003))
    hparams["momentum"]         =   float(hparam_file_dict.get("momentum", 0.9))
    hparams["weight_decay"]     =   float(hparam_file_dict.get("weight_decay", 0.0001))
    hparams["SAVE_PLOTS_EPOCH"] =   int(hparam_file_dict.get("SAVE_PLOTS_EPOCH", max([int(hparams["NUM_EPOCHS"]/10), 3])))
    hparams["SAVE_MODEL_EPOCH"] =   int(hparam_file_dict.get("SAVE_MODEL_EPOCH", max([int(hparams["NUM_EPOCHS"]/10), 3])))
    
    mapping = hparam_file_dict.get("MAPPING", None)
    if mapping is not None:
        hparams["MAPPING"] = json.loads(mapping)
    else:
        hparams["MAPPING"] = {'DUMMY': 0,'Fish': 1,'Cnidaria':2}

    hparams["DEVICE"] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    hparams["RESIZE_TO"] = 800
    hparams["SEED"] = 42
    hparams["IMAGE_DIRECTORY"] = "images"

    # initialize dataloaders
    train_loader, valid_loader = prepare_data_loaders(hparams)
    print("Number of training images: {}".format(len(train_loader)))
    print("Number of test images: {}".format(len(valid_loader)))

    # train
    start_training(train_loader, valid_loader, hparams)

    # make some outputs for debugging
    outdict = listEntriesRecursive("/opt/ml/")
    with open(hparams["OUT_DIR"]+"/mlDirectory.json", "a") as f:
        f.write(json.dumps(outdict))

    with open(hparams["HYPERPARAM_FILE"]) as f:
        text = f.read()

    with open(hparams["OUT_DIR"]+"/hparams.json", "a") as f:
        f.write(text)

    with open(hparams["DATA_CONF_FILE"]) as f:
        text = f.read()

    with open(hparams["OUT_DIR"]+"/confparams.json", "a") as f:
        f.write(text)