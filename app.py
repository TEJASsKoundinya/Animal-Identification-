
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='animal.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==89:
        preds="zebra"
    elif preds==88:
        preds="woodpecker"
    elif preds==87:
        preds="wombat"
    elif preds==86:
        preds="wolf"
    elif preds==85:
        preds="whale"
    elif preds==84:
        preds="turtle"
    elif preds==83:
        preds="turkey"
    elif preds==82:
        preds="tiger"
    elif preds==81:
        preds="swan"
    elif preds==80:
        preds="starfish"
    elif preds==79:
        preds="squirrel"
    elif preds==78:
        preds="squid"
    elif preds==77:
        preds="sparrow"
    elif preds==76:
        preds="snake"
    elif preds==75:
        preds="sheep"
    elif preds==74:
        preds="shark"
    elif preds==73:
        preds="seal"
    elif preds==72:
        preds="seahorse"
    elif preds==71:
        preds="sandpiper"
    elif preds==70:
        preds="rhinoceros"
    elif preds==69:
        preds="reindeer"
    elif preds==68:
        preds="rat"
    elif preds==67:
        preds="raccon"
    elif preds==66:
        preds="possum"
    elif preds==65:
        preds="porcupine"
    elif preds==64:
        preds="pigeon"
    elif preds==63:
        preds="pig"
    elif preds==62:
        preds="penguine"
    elif preds==61:
        preds="pelecaniformes"
    elif preds==60:
        preds="parrot"
    elif preds==59:
        preds="panda"
    elif preds==58:
        preds="oyster"
    elif preds==57:
        preds="ox"
    elif preds==56:
        preds="owl"
    elif preds==55:
        preds="otter"
    elif preds==54:
        preds="orangutan"
    elif preds==53:
        preds="okapi"
    elif preds==52:
        preds="octopus"
    elif preds==51:
        preds="mouse"
    elif preds==50:
        preds="moth"
    elif preds==49:
        preds="mosquito"
    elif preds==48:
        preds="lobster"
    elif preds==47:
        preds="lizard"
    elif preds==46:
        preds="lion"
    elif preds==45:
        preds="leopard"
    elif preds==44:
        preds="ladybugs"
    elif preds==43:
        preds="Koala"
    elif preds==42:
        preds="Kangaroo"
    elif preds==41:
        preds="jellyfish"
    elif preds==40:
        preds="hyena"
    elif preds==39:
        preds="hummingbird"
    elif preds==38:
        preds="hourse"
    elif preds==37:
        preds="hornbill"
    elif preds==36:
        preds="hipopotamus"
    elif preds==35:
        preds="hedgehog"
    elif preds==34:
        preds="hare"
    elif preds==33:
        preds="hamster"
    elif preds==32:
        preds="grasshopper"
    elif preds==31:
        preds="gorilla"
    elif preds==30:
        preds="goose"
    elif preds==29:
        preds="goldfish"
    elif preds==28:
        preds="goat"
    elif preds==27:
        preds="fox"
    elif preds==26:
        preds="fly"
    elif preds==25:
        preds="flamingo"
    elif preds==24:
        preds="elephant"
    elif preds==23:
        preds="eagle"
    elif preds==22:
        preds="duck"
    elif preds==21:
        preds="dragonfly"
    elif preds==20:
        preds="donky"
    elif preds==19:
        preds="dolphin"
    elif preds==18:
        preds="dog"
    elif preds==17:
        preds="deer"
    elif preds==16:
        preds="crow"
    elif preds==15:
        preds="crab"
    elif preds==14:
        preds="coyote"
    elif preds==13:
        preds="cow"
    elif preds==12:
        preds="cockroach"
    elif preds==11:
        preds="chimpanzee"
    elif preds==10:
        preds="caterpillar"
    elif preds==9:
        preds="cat"
    elif preds==8:
        preds="butterfly"
    elif preds==7:
        preds="boar"
    elif preds==6:
        preds="bison"   
    elif preds == 5:
        preds = "beetle"
    elif preds == 4:
        preds = "bee"
    elif preds == 3:
        preds = "bear"
    elif preds == 2:
        preds = "bat"
    elif preds == 1:
        preds = "badger"
    elif preds == 0:
        preds = "antelope"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('app.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)