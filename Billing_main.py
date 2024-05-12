#!/usr/bin/env python3

import cv2
from picamera2 import Picamera2
import os
import sys, getopt, time
import numpy as np
from edge_impulse_linux.image import ImageImpulseRunner
import serial

import requests
import json
from requests.structures import CaseInsensitiveDict

runner = None
old_label = None
old_weight = 0
post_done = 0


product_id = {'Tomato': 1, 'Monaco': 2, 'Lays': 3, 'Lemon': 4}
names = {'Tomato':'Tomato', 'Monaco':'Monaco', 'Lays':'Lays', 'Lemon':'Lemon'}
units = {'Tomato':'KG', 'Monaco':'U', 'Lays':'U', 'Lemon':'KG'}
unit_weight = {'Tomato':1, 'Monaco':1, 'Lays':1 , 'Lemon':1}
unit_price = {'Tomato':30, 'Monaco':10, 'Lays':5, 'Lemon':50}

def help():
    print('python3 main.py <path_to_model.eim>')

def find_weight():
    ser = serial.Serial('/dev/ttyACM0', 57600, timeout=1)
    weight = 0
    time.sleep(3)
    for i in range(50):
        weight += float(ser.readline().decode('utf-8').rstrip())
    return (weight/50)
        
    
def post_data_to_api(label, weight):
    
    if units[label]=='U':
        taken = round(weight/unit_weight[label])
    else:
        taken = round(weight/1000, 2)
        
    final_rate = taken * unit_price[label]
    
    url = "https://autobill-9xjt.onrender.com/product"
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    data_dict = {"id":str(product_id[label]),"name":names[label],"price":unit_price[label],"unit":units[label],"taken":str(taken),"payable":str(final_rate)}
    data = json.dumps(data_dict)
    resp = requests.post(url, headers=headers, data=data)
    print(resp.status_code)
    return (resp.status_code)

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) != 1:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1296,972)}))
    picam2.start()

    with ImageImpulseRunner(modelfile) as runner:
        try:
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            
            while True:
                global old_label, post_done, old_weight
                img = picam2.capture_array()
                if img is None:
                    print('Failed to capture image')
                    exit(1)
                cv2.imshow('image',img)
                # imread returns images in BGR format, so we need to convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                features, cropped = runner.get_features_from_image(img)

                res = runner.classify(features)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        cropped = cv2.rectangle(cropped, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
                        if bb['value'] > 0.9:
                            if bb['label'] == old_label:
                                weight = 1
                                if weight > old_weight:
                                    if (post_done == 0):
                                        print(bb['label'], weight)
                                        status = post_data_to_api(bb['label'], weight)
                                        if status == 200:
                                            post_done = 1
                                else:
                                    old_weight = weight
                                
                            else:
                                old_label = bb['label']
                                post_done = 0        

                # the image will be resized and cropped and displayed
                cv2.imshow('image', cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])