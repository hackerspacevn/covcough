# This lambda trigger on every createobject. Using this lambda, we can execute trained model against uploaded audio record and put result into /results folder under the same s3 bucket.

import boto3
import sys
import time
import os
import json
import re
from datetime import datetime
import urllib.request
from urllib.parse import urlsplit, urlunsplit


# All python libraries for ML and Sound processing library
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd

SLACK_URL = os.getenv('SLACK_WEBHOOK')
APIGATEWAY_LAMBDA = os.getenv('APIGATEWAY_LAMBDA')
DEBUG = os.getenv('DEBUG')


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

# Extract Melspectrogram function


def mel_specs(data, sr, nmels=64):
    mel = librosa.feature.melspectrogram(data, sr, n_mels=nmels)
    mel_db = librosa.power_to_db(mel)
    mel_db = padding(mel_db, nmels, nmels)
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

# Update the function on 14/04/2022
# Function to predict covid-19 for Multi-language interface


def prediction_COVID(lmodel, filename, person, nmels, language):
    # Thi???t l???p ng??n ng???
    if language == 'VN':
        file_quality = 'Ch???t l?????ng ti???ng ho ghi ??m ?????t {}% y??u c???u\n'
        quality_alert = 'B???n n??n ghi ??m l???i ????? k???t qu??? ch??nh x??c h??n!'
        columns_df = ['B??nh th?????ng', 'COVID-19']
        xlabel_df = 'S??? ti???ng ho\n'
        ylabel_df = 'X??c su???t (%)'
        set_title_df = 'Bi???u ????? ph??n t??ch ti???ng ho ???? ghi ??m'
        result_alert = 'Ti???ng ho c???a b???n gi???ng ng?????i nhi???m Covid-19\n Trung b??nh: {prob_mean}%  Cao nh???t: {prob_max}% (Trung v???: {prob_median}%)'
        labelnames = ['B??nh th?????ng', 'Covid-19']
        text_alert = ['{}%\n B??nh th?????ng', '{}%\n Covid-19']
        alert = ['B???n kh??ng c?? nguy c??!', 'B???n c???n theo d??i th??m!', 'B???n ??ang c?? nguy c??, c???n ti???p t???c theo d??i th??m!', 'B???n c?? nguy c?? cao, ????? ngh??? theo d??i th??m!',
                 'B???n c?? nguy c?? r???t cao, n??n s??? d???ng test nhanh ????? ki???m tra!', 'B???n n??n ki???m tra l???i test nhanh ho???c test PCR!', 'Xin vui l??ng ki???m tra l???i ch???t l?????ng ghi ??m ti???ng ho!']
        title_plot = 'K???t qu??? ph??n t??ch ti???ng ho'
        title1 = 'X??c su???t trung b??nh'
        title2 = 'X??c su???t l???n nh???t'
        alert_cough_file1 = 'Kh??ng c?? ti???ng ho trong t???p ghi ??m!'
        alert_cough_file2 = 'T???p ghi ??m {} b??? l???i!'
        num_cough_alert = 'S??? ti???ng ho ???????c ghi ??m l?? {} '
        num_cough_alert1 = '(N??n ho ??t nh???t 6 ti???ng ho)\n'
        num_cough_limit0 = 'Kh??ng c?? ti???ng ho n??o v?????t qu?? 50%!\n'
        num_cough_limit1 = 'Ch??? c?? {} ti???ng ho v?????t qu?? 50%!\n'
        num_cough_limit2 = 'C?? {} ti???ng ho v?????t qu?? 50%!\n'
        date_time_alert = '(th???c hi???n v??o l??c {} ng??y {} theo gi??? UTC)'
        alert_cough_covid = 'Ti???ng ho c???a b???n c?? kh??? n??ng gi???ng c???a ng?????i nhi???m Covid-19\n'
        alert_final = 'K???t lu???n:\n'
    if language == 'EN':
        file_quality = 'The quality of cough sounds in the recording file is {}%\n'
        quality_alert = 'The detection result is not accuracy because of noises in recording file!\n'
        columns_df = ['Healthy', 'COVID-19']
        xlabel_df = 'Number of cough\n'
        ylabel_df = 'Likelyhood (%)'
        set_title_df = 'Analysis chart of recorded cough sounds'
        result_alert = 'Likelyhood of your cough being similar to Covid-19 patient\n Average: {prob_mean}%  Highest: {prob_max}%  (Median: {prob_median}%)'
        labelnames = ['Healthy', 'Covid-19']
        text_alert = ['{}%\n Healthy', '{}%\n Covid-19']
        alert = ['You have no risk!', 'You have a risk!', 'You have a high risk!', 'You have a very high risk!',
                 'You should take a Rapid test or PCR test!', 'You should take a Rapid test or PCR test!', 'Please check the quality of cough recording!']
        title_plot = 'Coughing diagnosis result'
        title1 = 'Average probability'
        title2 = 'Maximum probability'
        alert_cough_file1 = 'There is no cough in the recording file!'
        alert_cough_file2 = 'There is an error!'
        num_cough_alert = 'Number of recorded cough sounds are {} '
        num_cough_alert1 = '(recommended at least six cough sounds)\n'
        num_cough_limit0 = 'There is no cough sound exceeding 50%!\n'
        num_cough_limit1 = 'There is only {} cough sounds exceeding 50%!\n'
        num_cough_limit2 = 'There are {} cough sounds exceeding 50%!\n'
        date_time_alert = '            (tested on {} {} UTC)'
        alert_cough_covid = 'Likelyhood of your cough being similar to Covid-19 patient\n'
        alert_final = 'Conclusion:\n'
    # L???y ng??y v?? gi???
    now = datetime.now()
    date_now = now.date().strftime('%d/%m/%Y')
    time_now = now.time().strftime('%H:%M:%S')
    date_time_alert = date_time_alert.format(time_now, date_now)
    # L???y MelSpec
    y, sr = librosa.load(filename, sr=None)
    cough_segments, cough_mask = segment_cough(
        y, sr, cough_padding=0.1, min_cough_len=0.05)
    pos1 = []
    pos2 = []
    # ?????m s??? ti???ng ho
    num_cough = len(cough_segments)
    if len(cough_segments) == 0:
        print(alert_cough_file1)
    else:
        for i in range(len(cough_segments)):
            # Check the length of cough more than 64 Mel filter band
            if len(cough_segments[i]) > 32256:
                a = cough_segments[i][:32256]
                melspec = mel_specs(a, sr, nmels)
                melspec = np.array([melspec.reshape((nmels, nmels, 1))])
                prob = lmodel.predict(melspec)
                pos1.append(prob[0]*100)
                pos2.append(prob[0][1]*100)
                a = cough_segments[i][32256:]
                melspec = mel_specs(a, sr, nmels)
                melspec = np.array([melspec.reshape((nmels, nmels, 1))])
                prob = lmodel.predict(melspec)
                pos1.append(prob[0]*100)
                pos2.append(prob[0][1]*100)
                num_cough = num_cough+1
            else:
                melspec = mel_specs(cough_segments[i], sr, nmels)
                melspec = np.array([melspec.reshape((nmels, nmels, 1))])
                prob = lmodel.predict(melspec)
                pos1.append(prob[0]*100)
                pos2.append(prob[0][1]*100)
        test_result = [np.mean(pos1), np.max(pos1)]
    # V??? k???t qu??? ph??n t??ch ti???ng ho
    result = pd.DataFrame(pos1, columns=columns_df)
    # T??ng index b???t ?????u t??? 1
    result.index = result.index+1
    ax1 = result.plot.bar(xlabel=xlabel_df+person, ylabel=ylabel_df,
                          color=['#00cacd', '#e73786'], rot=0, figsize=(7, 4))
    # Hi???u ch???nh legend b??n ngo??i ????? th???
    ax1.legend(bbox_to_anchor=(1.0, 1.0))
    plt.title(set_title_df, fontsize=16, loc='center')
    for p in ax1.patches:
        ax1.annotate(str(round(p.get_height(), 2)), (p.get_x() *
                     1.005, p.get_height() * 1.005), horizontalalignment='left')
# Hi???u ch???nh m??u theo t???ng c???nh b??o B??nh th?????ng<20%-M??u xanh, Nguy c?? 20-40% m??u v??ng, Nguy c?? cao 40-50% m??u cam
    test0 = round((np.mean(pos2)), 2)
    test1 = round((np.max(pos2)), 2)
    text_mean = text_alert[1]
    text_max = text_alert[1]
    if round(np.mean(pos2), 2) <= 20:
        colordisplay_mean = '#e73786'
        colordisplay_max = '#e73786'
        if round(np.max(pos2), 2) >= 50:
            text1 = alert[1]
        else:
            text1 = alert[0]
    if round(np.mean(pos2), 2) > 20 and round(np.mean(pos2), 2) <= 30:
        colordisplay_mean = '#e73786'
        colordisplay_max = '#e73786'
        if round(np.max(pos2), 2) >= 50:
            text1 = alert[3]
        else:
            text1 = alert[2]
    if round(np.mean(pos2), 2) > 30 and round(np.mean(pos2), 2) <= 40:
        colordisplay_mean = '#e73786'
        colordisplay_max = '#e73786'
        if round(np.max(pos2), 2) >= 50:
            text1 = alert[4]
        else:
            text1 = alert[3]
    if round(np.mean(pos2), 2) > 40 and round(np.mean(pos2), 2) < 50:
        colordisplay_mean = '#e73786'
        colordisplay_max = '#e73786'
        text1 = alert[4]
    if round(np.mean(pos2), 2) > 50:
        colordisplay_mean = '#e73786'
        colordisplay_max = '#e73786'
        text1 = alert[5]
    # Ki???m tra s??? ti???ng ho tr??n 50% covid-19
    num_cough50 = result[result['COVID-19'] >= 50].count()[1]
    # Th??ng b??o s??? ti???ng ho
    thongbao = num_cough_alert.format(num_cough)
    # Ki???m tra s??? ti???ng ho
    if num_cough < 6:
        thongbao = thongbao+num_cough_alert1
    else:
        thongbao = thongbao+'\n'
    # Thi???t l???p th??ng b??o k???t qu???
    if num_cough50 == 0:
        thongbao = thongbao+num_cough_limit0
    elif num_cough50 == 1:
        thongbao = thongbao+num_cough_limit1.format(num_cough50)
    elif num_cough50 == 2:
        thongbao = thongbao+num_cough_limit2.format(num_cough50)
    elif num_cough50 >= 3:
        thongbao = thongbao+num_cough_limit2.format(num_cough50)
        thongbao = thongbao+alert_cough_covid
    # V??? bi???u ????? k???t lu???n ti???ng ho
    fig, ax2 = plt.subplots(1, 2, figsize=(7, 5))
    fig.suptitle(title_plot, fontsize=16)
    fig.text(0.1, 0.03, thongbao, fontsize=12, color='#000000')
    fig.text(0.1, 0.03, text1, fontsize=12, color='#ff0000')
    fig.text(0.22, 0.90, date_time_alert, fontsize=12, color='#000000')
    wedgeprops = {'width': 0.4, 'edgecolor': '#ffffff', 'linewidth': 1}
    textprops = {"fontsize": 12}
    size1 = [round((100-np.mean(pos2)), 2), round(np.mean(pos2), 2)]
    size2 = [round((100-np.max(pos2)), 2), round(np.max(pos2), 2)]
    ax2[0].pie(size1, pctdistance=0.7, wedgeprops=wedgeprops,
               startangle=90, colors=['#00cacd', '#e73786'], textprops=textprops)
    ax2[0].set_title(title1, fontsize=14)
    ax2[0].text(0, 0, text_mean.format(test0), ha='center',
                va='center', fontsize=18, color=colordisplay_mean)
    ax2[1].pie(size2, pctdistance=0.7, wedgeprops=wedgeprops,
               startangle=90, colors=['#00cacd', '#e73786'], textprops=textprops)
    ax2[1].set_title(title2, fontsize=14)
    ax2[1].text(0, 0, text_max.format(test1), ha='center',
                va='center', fontsize=18, color=colordisplay_max)
    plt.tight_layout()
    return ax1, ax2, test_result, pos2


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
    subfolder = ""
    objkeyparts = objkey.split("/")
    if (len(objkeyparts) > 2):
        subfolder = "/".join(objkeyparts[1:len(objkeyparts)-1])+"/"
        log("subfolder: {}".format(subfolder))

    filename = tmppath+objkeyparts[-1]
    log("filename: {}".format(filename))
    pngresult1 = filename[:-4]+"_1.png"
    pngresult2 = filename[:-4]+"_2.png"
    jsonresult = filename[:-4]+".json"
    # csvresult = filename[:-4]+".csv"

    # Download wav file from s3 bucket
    s3 = boto3.client('s3')
    s3.download_file(bucket, objkey, filename)

    # Add the path of ML model
    print("loading model!")
    final_model = keras.models.load_model(
        'Early_CoughCovid_ResNet50_15_02_2022.hdf5')
    # Add entry name of person
    person_name = 'anonymous'
    print("Testing Covid!")
    img1, img2, listresult, prob = prediction_COVID(
        final_model, filename, filename, nmels=64, language='VN')
    f = open(jsonresult, 'w')
    stringresult = [str(i) for i in listresult]
    f.write(json.dumps({"Result": stringresult}))
    f.close()

    if (img1 != None):
        # prob_result = np.array(prob).tolist()
        # result_data = pd.DataFrame(prob_result, columns=[
        #                            'Healthy', 'Covid-19'])
        # result_data.to_csv(csvresult)
        # Save image
        img1.figure.savefig(pngresult1, bbox_inches='tight')
        plt.savefig(pngresult2, bbox_inches='tight')

        # Upload results
        s3.upload_file(pngresult1, bucket, "results/" +
                       subfolder+pngresult1.split("/")[-1])
        s3.upload_file(pngresult2, bucket, "results/" +
                       subfolder+pngresult2.split("/")[-1])
        # s3.upload_file(csvresult, bucket, "results/" +
        #                subfolder+csvresult.split("/")[-1])
        # Remove temp files
        os.remove(pngresult1)
        os.remove(pngresult2)
        # os.remove(csvresult)
    os.remove(filename)
    s3.upload_file(jsonresult, bucket, "results/" +
                   subfolder+jsonresult.split("/")[-1])
    os.remove(jsonresult)

    info = getobjmeta(bucket, objkey)
    info["resulturl1"] = APIGATEWAY_LAMBDA + \
        "/download/results/"+subfolder+pngresult1.split("/")[-1]
    info["resulturl2"] = APIGATEWAY_LAMBDA + \
        "/download/results/"+subfolder+pngresult2.split("/")[-1]
    info["sourceip"] = sourceip

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
    print("UserIP: " + sourceip)
    nofify_slack(payload)


# This is for local test
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    # Add the path of ML model
    print("loading model!")
    final_model = keras.models.load_model(
        './Early_CoughCovid_ResNet50_15_02_2022.hdf5')
    # Add file path here
    dir_path = './'
    filename = 'test.wav'
    # Add entry name of person
    person_name = 'anonymous'
    print("testing Covid!")
    # Select language='VN' for Vietnamese and language='EN' for English
    img1, img2, listresult, prob = prediction_COVID(
        final_model, dir_path+filename, filename, nmels=64, language='VN')
    # Save image 1 and 2
    img1.figure.savefig('result1.png', bbox_inches='tight')
    plt.savefig('result2.png', bbox_inches='tight')
    # Display result
    stringresult = [str(i) for i in listresult]
#   print(json.dumps(stringresult))
#   prob_result=np.array(prob).tolist()
#   result_data=pd.DataFrame(prob_result,columns=['Healthy','Covid-19'])
#   result_data.to_csv('result.csv')
