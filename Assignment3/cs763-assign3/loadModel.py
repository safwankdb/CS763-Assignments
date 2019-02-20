import argparse
import os
from Model import Model
from Layers import Linear, ReLU


def load():
    args = argparse.ArgumentParser()
    args.add_argument('-modelName')
    args.add_argument('-data')
    args.add_argument('-target')
    args = args.parse_args()

    if args.modelName:
        os.mkdir(args.modelName)

    model_filename = raw_input('Model File: ')
    model_file = open(model_filename, 'r')
    layer_arr = model_file.readlines()[:-2]
    layer_arr = [x[:-1] for x in layer_arr]
    model_file.close()

    layer_arr = layer_arr[1:]

    model = Model()

    for i in layer_arr:
        s = i.split()
        if s[0] == 'linear':
            layer = Linear(int(s[1]), int(s[2]))
            model.addLayer(layer)
        elif s[0] == 'relu':
            layer = ReLU()
            model.addLayer(layer)
    print 'Model loaded successfully from file:', model_filename
    return model
