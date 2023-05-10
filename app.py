import shutil
import sys
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt



# global b
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
   
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
       

        shutil.copy("C:\\Users\\ANJALI LADWA\\OneDrive\\Desktop\\CERVICAL_CANCER_Final\\CERVICAL_CANCER\\test\\"+fileName,dst)
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'cervicalcancer-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 6, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        accuracy=""
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            
            if np.argmax(model_out) == 0:
                str_label = 'stage1'
            elif np.argmax(model_out) == 1:
                str_label = 'stage2'
            elif np.argmax(model_out) == 2:
                str_label = 'stage3'
            elif np.argmax(model_out) == 3:
                str_label = 'stage4'

            elif np.argmax(model_out) == 4:
                str_label = 'stage5'
            elif np.argmax(model_out) == 5:
                str_label = 'Normal'
           
            
            
            
            if str_label == 'stage1':
                status = "stage1"
                print("The predicted image of the stage1 is with a accuracy of {} %".format(model_out[0]*100))
                accuracy = "The predicted image of the stage1 is with a accuracy of {} %".format(model_out[0]*100)
                display= " A cone biopsy is the preferred procedure for women who want to have children after the cancer is treated. A simple hysterectomy may be an option if the cancer shows no lymphovascular invasion and the edges of the biopsy have no cancer cells. "
                
            
            elif str_label == 'stage2':
                status = "stage2"
                print("The predicted image of the stage2 is with a accuracy of {} %".format(model_out[1]*100))
                accuracy = "The predicted image of the stage2 is with a accuracy of {} %".format(model_out[1]*100)
                display="External beam radiation therapy (EBRT) to the pelvis plus brachytherapy. Radical hysterectomy with removal of pelvic lymph nodes"

                
                
            elif str_label == 'stage3':
                status = "stage3"
                print("The predicted image of the stage3 is with a accuracy of {} %".format(model_out[2]*100))
                accuracy = "The predicted image of the stage3 is with a accuracy of {} %".format(model_out[2]*100)
                display= "Radiation with or without chemotherapy: The radiation therapy includes both external beam radiation and brachytherapy. The chemo may be cisplatin, carboplatin, or cisplatin plus fluorouracil. "


            elif str_label == 'stage4':
                status = "stage4"
                print("The predicted image of the stage4 is with a accuracy of {} %".format(model_out[3]*100))
                accuracy = "The predicted image of the stage4 is with a accuracy of {} %".format(model_out[3]*100)
                display="Chemoradiation: The chemo may be cisplatin, carboplatin, or cisplatin plus fluorouracil. The radiation therapy includes both external beam radiation and brachytherapy."

            elif str_label == 'stage5':
                status = "stage5"
                print("The predicted image of the stage5 is with a accuracy of {} %".format(model_out[4]*100))
                accuracy = "The predicted image of the stage5 is with a accuracy of {} %".format(model_out[4]*100)
                display=" Stage IVB cervical cancer is not usually considered curable. standard chemo regimens include a platinum drug (cisplatin or carboplatin) along with another drug such as paclitaxel (Taxol), gemcitabine (Gemzar), or topotecan."
                
                     
            elif str_label == 'Normal':
                status= 'Normal'
                print("The predicted image of the Normal is with a accuracy of {} %".format(model_out[5]*100))
                accuracy = "The predicted image of the Normal is with a accuracy of {} %".format(model_out[5]*100)
                #print(" this is normal case No need to worry!")
                display = " This is normal case No need to worry!"

            

            
                
            return render_template('home.html', status=str_label,accuracy=accuracy,display=display,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
    
        return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
