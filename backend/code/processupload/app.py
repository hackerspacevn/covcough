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

# Update the function on 03/04/2022
# Function to predict covid-19 for Multi-language interface


def prediction_COVID(lmodel, filename, person, nmels, language):
    # Thiết lập ngôn ngữ
    if language == 'VN':
        file_quality = 'Chất lượng tiếng ho ghi âm đạt {}% yêu cầu\n'
        quality_alert = 'Bạn nên ghi âm lại để kết quả chính xác hơn!'
        columns_df = ['Bình thường', 'COVID-19']
        xlabel_df = 'Số tiếng ho\n'
        ylabel_df = 'Xác suất (%)'
        set_title_df = 'Biểu đồ phân tích tiếng ho đã ghi âm'
        result_alert = 'Tiếng ho của bạn giống người nhiễm Covid-19\n Trung bình: {prob_mean}%  Cao nhất: {prob_max}% (Trung vị: {prob_median}%)'
        labelnames = ['Bình thường', 'Covid-19']
        text_alert = ['{}%\n Bình thường', '{}%\n Covid-19']
        alert = ['Bạn không có nguy cơ!', 'Bạn cần theo dõi thêm!', 'Bạn đang có nguy cơ, cần tiếp tục theo dõi thêm!', 'Bạn có nguy cơ cao, đề nghị theo dõi thêm!',
                 'Bạn có nguy cơ rất cao, nên sử dụng test nhanh để kiểm tra!', 'Bạn nên kiểm tra lại test nhanh hoặc test PCR!', 'Xin vui lòng kiểm tra lại chất lượng ghi âm tiếng ho!']
        title_plot = 'Kết quả phân tích tiếng ho'
        title1 = 'Xác suất trung bình'
        title2 = 'Xác suất lớn nhất'
        alert_cough_file1 = 'Không có tiếng ho trong tệp ghi âm!'
        alert_cough_file2 = 'Tệp ghi âm {} bị lỗi!'
        num_cough_alert = 'Số tiếng ho được ghi âm là {} '
        num_cough_alert1 = '(Nên ho ít nhất 6 tiếng ho)\n'
        num_cough_limit0 = 'Không có tiếng ho nào vượt quá 70%!\n'
        num_cough_limit1 = 'Chỉ có {} tiếng ho vượt quá 70%!\n'
        num_cough_limit2 = 'Có {} tiếng ho vượt quá 70%!\n'
        alert_cough_covid = 'Tiếng ho của bạn có khả năng giống của người nhiễm Covid-19\n'
        alert_final = 'Kết luận:\n'
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
        num_cough_limit0 = 'There is no cough sound exceeding 70%!\n'
        num_cough_limit1 = 'There is only {} cough sounds exceeding 70%!\n'
        num_cough_limit2 = 'There are {} cough sounds exceeding 70%!\n'
        alert_cough_covid = 'Likelyhood of your cough being similar to Covid-19 patient\n'
        alert_final = 'Conclusion:\n'
    # Lấy MelSpec
    y, sr = librosa.load(filename, sr=None)
    cough_segments, cough_mask = segment_cough(
        y, sr, cough_padding=0.1, min_cough_len=0.05)
    pos1 = []
    pos2 = []
    # Đếm số tiếng ho
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

    # Vẽ kết quả phân tích tiếng ho
    result = pd.DataFrame(pos1, columns=columns_df)
    # Tăng index bắt đầu từ 1
    result.index = result.index+1
    ax1 = result.plot.bar(xlabel=xlabel_df+person, ylabel=ylabel_df,
                          color=['#00cacd', '#e73786'], rot=0, figsize=(7, 4))
    # Hiệu chỉnh legend bên ngoài đồ thị
    ax1.legend(bbox_to_anchor=(1.0, 1.0))
    plt.title(set_title_df, fontsize=16, loc='center')
    for p in ax1.patches:
        ax1.annotate(str(round(p.get_height(), 2)), (p.get_x() *
                     1.005, p.get_height() * 1.005), horizontalalignment='left')

# Hiệu chỉnh màu theo từng cảnh báo Bình thường<20%-Màu xanh, Nguy cơ 20-40% màu vàng, Nguy cơ cao 40-50% màu cam
    test0 = round((np.mean(pos2)), 2)
    test1 = round((np.max(pos2)), 2)
    text_mean = text_alert[1]
    text_max = text_alert[1]
    if round(np.mean(pos2), 2) <= 20:
        colordisplay_mean = '#00cacd'
        colordisplay_max = '#00cacd'
        text_mean = text_alert[0]
        text_max = text_alert[0]
        test0 = round((100-np.mean(pos2)), 2)
        if round(np.max(pos2), 2) >= 50:
            text1 = alert[1]
            text_max = text_alert[1]
            test1 = round((np.max(pos2)), 2)
            colordisplay_max = '#e73786'
        else:
            text1 = alert[0]
            test1 = round((100-np.max(pos2)), 2)

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

    # Kiểm tra số tiếng ho trên 50% covid-19
    num_cough50 = result[result['COVID-19'] >= 70].count()[1]
    # Thông báo số tiếng ho
    thongbao = num_cough_alert.format(num_cough)
    # Kiểm tra số tiếng ho
    if num_cough < 6:
        thongbao = thongbao+num_cough_alert1
    else:
        thongbao = thongbao+'\n'
    # Thiết lập thông báo kết quả
    if num_cough50 == 0:
        thongbao = thongbao+num_cough_limit0
    elif num_cough50 == 1:
        thongbao = thongbao+num_cough_limit1.format(num_cough50)
    elif num_cough50 == 2:
        thongbao = thongbao+num_cough_limit2.format(num_cough50)
    elif num_cough50 >= 3:
        thongbao = thongbao+num_cough_limit2.format(num_cough50)
        thongbao = thongbao+alert_cough_covid
    # Vẽ biểu đồ kết luận tiếng ho
    fig, ax2 = plt.subplots(1, 2, figsize=(7, 5))
    fig.suptitle(title_plot, fontsize=16)
    fig.text(0.1, 0.07, thongbao, fontsize=12, color='#000000')
    fig.text(0.1, 0.07, text1, fontsize=12, color='#ff0000')
    wedgeprops = {'width': 0.4, 'edgecolor': '#ffffff', 'linewidth': 1}
    textprops = {"fontsize": 12}
    size1 = [round((100-np.mean(pos2)), 2), round(np.mean(pos2), 2)]
    size2 = [round((100-np.max(pos2)), 2), round(np.max(pos2), 2)]
    ax2[0].pie(size1, pctdistance=0.7, wedgeprops=wedgeprops,
               startangle=90, colors=['#00cacd', '#e73786'], textprops=textprops)
    ax2[0].set_title(title1, fontsize=12)
    ax2[0].text(0, 0, text_mean.format(test0), ha='center',
                va='center', fontsize=16, color=colordisplay_mean)
    ax2[1].pie(size2, pctdistance=0.7, wedgeprops=wedgeprops,
               startangle=90, colors=['#00cacd', '#e73786'], textprops=textprops)
    ax2[1].set_title(title2, fontsize=12)
    ax2[1].text(0, 0, text_max.format(test1), ha='center',
                va='center', fontsize=16, color=colordisplay_max)
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
    final_model = keras.models.load_model('Early_CoughCovid_ResNet50_15_02_2022.hdf5')
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
        img1.figure.savefig(pngresult1,bbox_inches='tight')
        plt.savefig(pngresult2,bbox_inches='tight')

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
    print(sourceip)
    nofify_slack(payload)


# This is for local test
if __name__ == "__main__":
  tf.get_logger().setLevel('ERROR')
  tf.autograph.set_verbosity(3)
  #Add the path of ML model
  print("loading model!")
  final_model = keras.models.load_model('./Early_CoughCovid_ResNet50_15_02_2022.hdf5')
  #Add file path here
  dir_path='./'
  filename='test.wav'
  #Add entry name of person
  person_name='anonymous'
  print("testing Covid!")
  #Select language='VN' for Vietnamese and language='EN' for English
  img1, img2, listresult, prob = prediction_COVID(final_model,dir_path+filename,filename,nmels=64,language='VN')
  #Save image 1 and 2
  img1.figure.savefig('result1.png',bbox_inches='tight')
  plt.savefig('result2.png',bbox_inches='tight')
  #Display result
  stringresult=[str(i) for i in listresult]
#   print(json.dumps(stringresult))
#   prob_result=np.array(prob).tolist()
#   result_data=pd.DataFrame(prob_result,columns=['Healthy','Covid-19'])
#   result_data.to_csv('result.csv')