# Captcha Classifier on Raspberry pi
To classify variable length captchas using a tensorflow lite model.

## Create a virtual environment
 $ python3 -m venv /path/to/new/virtual/env

## Install the packages
 $ pip install -r requirements.txt

## Download the captcha set
 $ python3 download_captchas.py

## Generate the validation set
 $ python3 generate.py --width 128 --height 64 --length 6 --symbols symbols_generate.txt --count 13000 --output-dir validation_captcha

## Generate the training set
  $ python3 generate.py --width 128 --height 64 --length 6 --symbols symbols_generate.txt --count 150000 --output-dir training_captcha

## To classify the captcha set using model trained on local and converted to tflite

  $ python3 classify_tflite.py --model-name model3.tflite --captcha-dir muvazima_captchas/ --output model3_tflite_output.txt --symbols symbols.txt

 ## Output file
  $ cat model3_tflite_output.csv


