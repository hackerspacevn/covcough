# This lambda trigger on every createobject. Using this lambda, we can execute trained model against uploaded audio record and put result into /results folder under the same s3 bucket.

import boto3
import sys
import time
import os
import json
import re
import datetime
import urllib.request
from urllib.parse import urlsplit, urlunsplit


# All python libraries for ML and Sound processing library
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

SLACK_URL         = os.getenv('SLACK_WEBHOOK')
APIGATEWAY_LAMBDA = os.getenv('APIGATEWAY_LAMBDA')
DEBUG             = os.getenv('DEBUG')

def nofify_slack(payload):
    headers = {'Content-type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(SLACK_URL, data, headers)
    resp = urllib.request.urlopen(req)
    return resp

def log(msg):
    if (DEBUG == None):
        return
    print(msg)

def getobjmeta(bkt, key):
    log(bkt)
    log(key)
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket=bkt, Key=key)
    # exp = datetime.datetime.strptime(expstr,'%d %b %Y %H:%M:%S %Z')
    objsize = response['ContentLength']
    objname = ""
    try:
        filemetadata = json.loads(
            response['ResponseMetadata']['HTTPHeaders']['x-amz-meta-tag'])
        objname = filemetadata["name"]
    except:
        objname = "unknown-file-name"
    return {
        "downloadurl": APIGATEWAY_LAMBDA+"/download/"+key,
        "tag_filename": objname,
    }


# All functions
# segment_cough function is created by COUGHVID project (https://c4science.ch/diffusion/10770)
def segment_cough(x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power

    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator

    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""

    cough_mask = np.array([False]*len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier*rms

    # Segment coughs
    coughSegments = []
    padding = round(fs*cough_padding)
    min_cough_samples = round(fs*min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01*fs)
    below_th_counter = 0

    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding > min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end = i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding > min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i-padding if (i-padding >= 0) else 0
                cough_in_progress = True

    return coughSegments, cough_mask

# prediction_COVID function uses ML model to generate the prediction result (negative or positive)
def prediction_COVID(lmodel,filename,person,nmels=64):
    #Load the audio file
    y,sr=librosa.load(filename)
    cough_segments, cough_mask = segment_cough(y,sr,cough_padding=0.1,min_cough_len=0.05)
    pos1=[]
    pos2=[]
    test_result=[]
    if len(cough_segments)==0:
        test = 0
        ax = None
        ps2 = None
        text = 'No coughing sound detected'
    else:
        for i in range(len(cough_segments)):
            melspec=mel_specs(cough_segments[i],sr)
            melspec=np.array([melspec.reshape((nmels,nmels,1))])
            prob=lmodel.predict(melspec)
            pos1.append(prob[0][1])
            pos2.append(prob[0])
        test_result=[np.mean(pos1),np.max(pos1)]

        try:
            #Plot the result of each cough sound
            result=pd.DataFrame(pos2,columns=['Healthy','COVID-19'])
            #Tăng index bắt đầu từ 1
            result.index=result.index+1
            #Hiệu chỉnh legend bên ngoài đồ thị
            ax=result.plot.bar(xlabel='Số tiếng ho\n'+person,ylabel='Xác suất (%)',color=['limegreen','red'],rot=0,figsize=(10, 6))
            ax.legend(bbox_to_anchor=(1.0, 1.0))
            ax.set_title('Kết quả phân tích tiếng ho của bạn\n Trung bình: {prob_mean}%     Cao nhất: {prob_max}%'
            .format(prob_mean=round(np.mean(pos1)*100,2),prob_max=round(np.max(pos1)*100,2)))
            for p in ax.patches:
                ax.annotate(str(round(p.get_height()*100,2)), (p.get_x() * 1.005, p.get_height() * 1.005),horizontalalignment='left')
        except Exception as e: 
            print(e)
            test = 0
            ax = None
            ps2 = None
            text = 'No coughing sound detected'
    return ax, test_result, pos2

# Extract Melspectrogram function
def mel_specs(data, sr, nmels=64):
    mel = librosa.feature.melspectrogram(data, sr, n_mels=nmels)
    mel_db = librosa.power_to_db(mel)
    mel_db = padding(mel_db, 64, 64)
    return mel_db

# padding function to fit shape of array


def padding(array, xx, yy):
    h = array.shape[0]
    w = array.shape[1]
    a = (xx-h)//2
    aa = xx-a-h
    b = (yy-w)//2
    bb = yy-b-w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def processsample():
    local_path = ""

# Handler for s3 create object event!


def lambda_handler(event, context):
    tmppath = "/tmp/"
    bucket = event['Records'][0]['s3']['bucket']['name']
    objkey = event['Records'][0]['s3']['object']['key']
    try:
        sourceip = event['Records'][0]['requestParameters']['sourceIPAddress'] 
    except:
        sourceip = "N/A"

    # Check if there are subfolders between "/reports/{subfolder}/filename"
    # This way we can track a single user using subfoldername and let them submit multiple samples into same folder.
    log(objkey)
    subfolder=""
    objkeyparts = objkey.split("/")
    if (len(objkeyparts) > 2):
        subfolder="/".join(objkeyparts[1:len(objkeyparts)-1])+"/"
        log("subfolder: {}".format(subfolder))
    
    filename = tmppath+objkeyparts[-1]
    log("filename: {}".format(filename))
    pngresult=filename[:-4]+".png"
    jsonresult=filename[:-4]+".json"
    csvresult=filename[:-4]+".csv"

    # Download wav file from s3 bucket
    s3 = boto3.client('s3')
    s3.download_file(bucket,objkey,filename)

    # Add the path of ML model
    print("loading model!")
    final_model = keras.models.load_model(
        'Early_alexnet64x64_AIVN_val_loss.hdf5')
    # Add entry name of person
    person_name = 'anonymous'
    print("Testing Covid!")
    img, listresult, prob = prediction_COVID(final_model, filename, filename, nmels=64)
    f=open(jsonresult,'w')
    stringresult=[str(i) for i in listresult]
    f.write(json.dumps({"Result":stringresult}))
    f.close()

    if (img != None):
        prob_result=np.array(prob).tolist()
        result_data=pd.DataFrame(prob_result,columns=['Healthy','Covid-19'])
        result_data.to_csv(csvresult)
        # Save image
        plt.savefig(pngresult)
        # Upload results
        s3.upload_file(pngresult,bucket,"results/"+subfolder+pngresult.split("/")[-1])
        s3.upload_file(csvresult,bucket,"results/"+subfolder+csvresult.split("/")[-1])
        # Remove temp files
        os.remove(pngresult)
        os.remove(csvresult)
    os.remove(filename)
    s3.upload_file(jsonresult,bucket,"results/"+subfolder+jsonresult.split("/")[-1])
    os.remove(jsonresult)

    info = getobjmeta(bucket,objkey)
    info["resulturl"]= APIGATEWAY_LAMBDA+"/download/results/"+subfolder+pngresult.split("/")[-1]
    info["sourceip"]=sourceip
    
    msg = '''
A new record was received. This file will expire in 10 days:
```
{}
```
    '''.format(json.dumps(info, indent=4, sort_keys=True))

    payload = {
        "icon_emoji": ":helmet_with_white_cross:",
        "username": "covcough",
                    "text": msg
    }
    nofify_slack(payload)


# This is for local test
if __name__ == "__main__":
  #Add the path of ML model
  print("loading model!")
  final_model = keras.models.load_model('./Early_alexnet64x64_AIVN_val_loss.hdf5')
  #Add file path here
  dir_path='./'
  filename='test.wav'
  #Add entry name of person
  person_name='anonymous'
  print("testing Covid!")
  img,listresult,prob=prediction_COVID(final_model,dir_path+filename,filename,nmels=64)
  #Save image
  plt.savefig('result.png')
  #Display result
  stringresult=[str(i) for i in listresult]
  print(json.dumps(stringresult))
  prob_result=np.array(prob).tolist()
  result_data=pd.DataFrame(prob_result,columns=['Healthy','Covid-19'])
  result_data.to_csv('result.csv')
