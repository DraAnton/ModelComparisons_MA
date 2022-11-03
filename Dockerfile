# Use nvidia/cuda image
#FROM nvidia/cuda:11.7.1-base-ubuntu20.04
FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-training:1.6.0-gpu-py36-cu101-ubuntu16.04

WORKDIR /app

ENV DATA_DIR=/opt/ml/input/data/training
ENV DATA_CONF_FILE=/opt/ml/input/config/inputdataconfig.json
ENV HYPERPARAM_FILE=/opt/ml/input/config/hyperparameters.json
ENV OUT_DIR=/opt/ml/model

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
COPY utils utils
COPY ImageEnhancement ImageEnhancement
COPY FasterRCNN.py FasterRCNN.py

ENTRYPOINT ["python", "FasterRCNN.py"]

#RUN git clone https://github.com/DraAnton/ModelComparisons_MA.git
#WORKDIR /ModelComparisons_MA



