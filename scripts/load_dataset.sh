#!/bin/bash

pip install kaggle
WORKING_DIR=$(pwd)
mkdir -p $WORKING_DIR/data/raw
mkdir -p $WORKING_DIR/data/processed
cd $WORKING_DIR/data/raw
kaggle datasets download -d unclesamulus/blood-cells-image-dataset
unzip blood-cells-image-dataset.zip
rm blood-cells-image-dataset.zip
