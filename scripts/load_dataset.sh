#!/bin/bash

pip install kaggle
cd app/data/raw
kaggle datasets download -d unclesamulus/blood-cells-image-dataset
unzip blood-cells-image-dataset.zip
rm blood-cells-image-dataset.zip