This repo is based on the BEiT-3 repo which can be found [here](https://github.com/microsoft/unilm/tree/master/beit3)

# Setup Instructions
1. Clone this repo
2. Download the data and annotation files [here](https://gtvault-my.sharepoint.com/:u:/g/personal/ghaglund3_gatech_edu/ERiMwAtBB3dNgwJ425gAjGwBig4EeHV9PPLk88i1hgYXEQ?e=P23JUX)
3. Place the annotation files and img directory in the hm_data directory
4. Download the pretrained base BEiT-3 model [here](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_base_patch16_224.pth) and/or the large model [here](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/pretraining/beit3_large_patch16_224.pth)
5. Place the pretrained models in the models/pretrained directory
6. Make sure you are in the beit3 directory and run the following commands. These will set up a docker environment and install the needed dependencies:
```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel bash
apt-get update
apt-get install libgl1-mesa-glx
apt-get install libglib 2.0-0
pip install -r requirements.txt --use-feature=2020-resolver
```

## Build dataset files
Before training the model, run `python make_dataset.py` to generate the appropriate data files.

## Train Models
To finetune the models, run `python train_models.py`. You can edit this file to change which models to train and how to train them. This will also generate predictions on the test set

## Evaluate models
To calculate accuracy and AUROC on the generated predictions, run `python eval_models.py`. This will print the metrics and write them to results/final_results.json

## Graph training statistics
To graph statistics from the training of the models, run `python graph_logs.py`. This generates graphs in the graphs directory.
