#!/bin/bash

pip install kaggle
mkdir -p src/data/raw
mkdir -p src/data/processed
cd src/data/raw
kaggle datasets download -d unclesamulus/blood-cells-image-dataset
unzip blood-cells-image-dataset.zip
rm blood-cells-image-dataset.zip