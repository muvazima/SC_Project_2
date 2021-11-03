#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
start_time = time.time()

import os
import cv2
import numpy
import string
import random
import argparse
import pandas as pd
import tflite_runtime.interpreter as tflite

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def clean(original_image):
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    clean_img = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel)

    return clean_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
#    with tflite.device('/cpu:0'):
    with open(args.output, 'w') as output_file:

        with open(args.model_name, 'rb') as fid:
            tflite_model = fid.read()
        interpreter = tflite.Interpreter(args.model_name)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        filenames=[]
        predictions=[]
        for x in os.listdir(args.captcha_dir):
                # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            input_shape = input_details[0]['shape']
            input_data = numpy.array(image, dtype=numpy.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            label=''
            for char in range(6):
                output_data = numpy.squeeze(interpreter.get_tensor(output_details[char]['index']))
                label=label+(captcha_symbols[numpy.argmax(output_data)])

            # print(label)

            #prediction = model.predict(image)
            label=label.strip()
            label=label.replace(' ', '')
            label=label.replace('&','')
            filenames.append(x)
            predictions.append(label)
            output_file.write(x + ", " + label + "\n")

            print('Classified ' + x + ': '+label)

        df=pd.DataFrame(list(zip(filenames,predictions)))
        df=df.sort_values(df.columns[0], ascending = True)
        df.set_index(0, inplace=True)
        df.to_csv(args.output[:-4]+'.csv')

    print("Execution time: %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()
