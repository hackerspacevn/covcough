import boto3
import sys
import hashlib
import time
import secrets
import os
import json
import datetime
import re
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from botocore.client import Config

# Change BUCKET_NAME to your bucket name and
# KEY_NAME to the name of a file in the directory where you'll run the curl command.
# This backend can be easily tweaked to support multiple regions to conform with GDPR data storage requirement


bkt = os.getenv('BUCKETNAME',None)
awsregion = "ap-southeast-1"
seed = os.environ['SEED']
appurl = os.environ['APPURL']
signingkey = os.environ['SIGNINGKEY']

## This is a temporary solution to let administrators to generate signed URLs for Covid19 patients.
adminsecret = os.getenv('ADMINSECRET','localtest')

# Set the max object size.. (200mb)
maxobjectsize = 200000000

# verify hash sent from user
def verifyhash(msg,verifyhash,digest_size=16):
    h = hashlib.blake2b(key=bytearray(signingkey.encode()), digest_size=digest_size)
    h.update(bytearray(msg.encode()))
    return h.hexdigest()==verifyhash

# create hash value for given message
def createhash(msg,digest_size=16):
    h = hashlib.blake2b(key=bytearray(signingkey.encode()), digest_size=digest_size)
    h.update(bytearray(msg.encode()))
    return h.hexdigest()

# getposturl generate request to create a pre-signed POST for uploading data to S3 for recording
def getposturl(filename, subfolder=""):
    if (subfolder != "" and subfolder[-1] != "/"):
            subfolder = str(subfolder)+"/"
    print("Generate post url!")
    s3 = boto3.client('s3',endpoint_url='https://s3.ap-southeast-1.amazonaws.com',config=Config(region_name=awsregion, signature_version='s3v4'))
    fields = {
            "acl": "private",
            }
    conditions = [
        {"acl": "private"},
        {"content-type":"audio/wav"},
        ["content-length-range", 1, maxobjectsize],
        ["starts-with", "$x-amz-meta-tag", ""]
    ]

    keyname = "records/{}{}.wav".format(subfolder,filename)
    resp={
        "signedupload":s3.generate_presigned_post(Bucket=bkt,Key=keyname,Fields=fields,Conditions=conditions),
        "pngresult1": getobj("results/{}{}_1.png".format(subfolder,filename)),
        "pngresult2": getobj("results/{}{}_2.png".format(subfolder,filename)),
        "jsonresult": getobj("results/{}{}.json".format(subfolder,filename)),
        "csvresult": getobj("results/{}{}.csv".format(subfolder,filename))
    }
    

    return resp

# getobj generate link to download /results/{sha256}.img
def getobj(key):
    s3 = boto3.client('s3', config=Config(region_name=awsregion, signature_version='s3v4'))
    return {
        "signedurl": s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bkt,
            'Key': key},
            ExpiresIn=3600
        )
    }


def getuniqueid():
    # generate a random id to prevent bruteforce ;).
    random = seed+str(time.time())+str(secrets.randbits(256))
    h = hashlib.sha256()
    h.update(random.encode("utf-8"))
    return h.hexdigest()

def getindividualurl(uniqueid,status):
    # Check the individual bucket. If there are previous entries, move up the current sample count by 1
    s3 = boto3.client('s3', config=Config(region_name=awsregion, signature_version='s3v4'))
    objs=s3.list_objects(Bucket=bkt,Delimiter="/",Prefix="records/{}/".format(uniqueid))
    nextsample=1
    try:
        if ("Contents" in objs):
            for obj in objs["Contents"]:      
                filename = obj["Key"].split("/")[-1]
                sameplenumber = int(re.findall(r'\d+', filename)[0])
                if nextsample <= sameplenumber:
                    nextsample = sameplenumber + 1
    except Exception as e:
        print(e)
        pass
    print("Next sample is {}".format(nextsample))
    filename="sample{}_{}".format(str(nextsample),status)
    return getposturl(filename,subfolder=uniqueid)


def gettimetokenurl(status):
    filename=getuniqueid()
    filename = filename+"_"+status
    print("Generate filename {}".format(filename))
    return getposturl(filename)

## generate a token valid for n number of hours
def generatetimetoken(n):
    exp=int(time.time())+3600*n
    token="exp:{}".format(str(exp))
    signedhash=createhash(token)
    return {"timetoken":"{}.{}".format(token,signedhash)}

## generate n number of individual tokens for administrator to hand out to covid patients.
def generateindividualtokens(n, subfolder):
    result=[]
    for i in range(0,n):
        uniqueid=getuniqueid()
        token="id:{}".format(subfolder+uniqueid)
        signedhash=createhash(token)
        result.append("{}.{}".format(token,signedhash))
    return result

# main lambda app_handler function. see this url for references
#https://www.serverless.com/framework/docs/providers/aws/events/apigateway/#example-lambda-proxy-event-default
def app_handler(event, context):
    global bkt
    global awsregion
    print ("Starting lambda")
    try:
        referer = event["headers"]["Referer"]
    except Exception as e: 
        print(e)
        referer = ""
    path = event["path"]
    # In prod, we will exit and return 200ok
    if (appurl != "devmode" and not referer.startswith(appurl)):
        return {
        "statusCode": 200,
        "body"  : 'ok'
    }   
    split_url = urlsplit(referer)
    clean_path = split_url.scheme+"://"+split_url.netloc 
    print("App runs at {} and bucket is {}".format(clean_path,bkt))
    headers = {
        'Access-Control-Allow-Origin': clean_path,
        'Content-Type': "application/json"
    }
    statuscode = 404
    body = {"status_code":404}

    if path.startswith("/upload"):
         # By default, the link is expired
        expiredurl = True
        signeddata = ""
        if (signingkey == "none"):
            expiredurl = False
        elif (event["queryStringParameters"] != None and "token" in event["queryStringParameters"]):
            try:
                keyparameter = event["queryStringParameters"]["token"].split(".")
                signeddata      = keyparameter[0]
                signature = keyparameter[1]
                if (len(signature) == 16 or len(signature) == 32):
                    if (verifyhash(signeddata,signature,digest_size=int(len(signature)/2))):
                        expiredurl = False
                else: 
                    expiredurl = True
            except Exception as e:
                print(e)
                expiredurl = True
        # If key verification failed or not provided, throw and error
        if (expiredurl):
            return {
                "statusCode": 403,
                "headers": headers,
                "body"  : json.dumps({"err":"Invalid exp value"})
            }  
        # signeddata is legit.. let's do something with it
        # We have 2 scenarios: an unique user with an unique link and an user with time expire link.
        ## /upload/positive?token=exp:1645454545.{signingkey}
        ## /upload/negative?token=id:{unique user id}.{signingkey}
        status = path[8:].strip()
        if (status in ["positive","negative","unknown","demosite"]): 
            print("fileupload request received")
            try:
                if signeddata.startswith("exp"):
                    t = int(signeddata.split(":")[1].encode('utf-8'))
                    if (int(time.time()) < t):
                        body = gettimetokenurl(status)
                        statuscode = 200
                    else:
                        return {
                            "statusCode": 403,
                            "headers": headers,
                            "body"  : json.dumps({"err":"Time token expired"})
                        }
                elif signeddata.startswith("id"):
                    userid=signeddata.split(":")[1]
                    body = getindividualurl(userid,status)
                    statuscode = 200
            except Exception as e: 
                print("Some thing went wrong...")
                print(e)
                pass
    ## /admin/getindividual/n generate n number of individual tokens for Covid19 F0 patients.
    elif path.startswith("/admin/getindividual/"):
        if(event["queryStringParameters"] != None and "secret" in event["queryStringParameters"]):
            secret = event["queryStringParameters"]["secret"]
            if secret == adminsecret:
                if (("f0" in event["queryStringParameters"]) and (event["queryStringParameters"]["f0"].lower().strip() == "true")):
                    numberoftoken=abs(int(path[21:]))
                    body = generateindividualtokens(numberoftoken, "study/")
                    statuscode = 200
                else:
                    numberoftoken=abs(int(path[21:]))
                    body = generateindividualtokens(numberoftoken, "")
                    statuscode = 200

    ## /admin/gettimetoken/n generate a token that expire in n hours.
    elif path.startswith("/admin/gettimetoken/"):
        if(event["queryStringParameters"] != None and "secret" in event["queryStringParameters"]):
            secret = event["queryStringParameters"]["secret"]
            if secret == adminsecret:
                numberofhours=abs(int(path[20:]))
                body = generatetimetoken(numberofhours)
                statuscode = 200
    ## getfile return a signed S3 url in a json response.
    elif path.startswith("/getfile"):
        print("getfile request received")
        try:
            body = getobj(path[9:]) 
            statuscode = 200
        except Exception as e: 
            print(e)
            body = {"status_code":404}
            statuscode = 404 
    ## download reidrect user immediately to the signed S3 url
    elif path.startswith("/download"):
        print("download request received")
        try:
            s3obj = getobj(path[10:]) 
            headers["Location"]=s3obj["signedurl"]
            statuscode = 301
        except Exception as e: 
            print(e)
            body = {"status_code":404}
            statuscode = 404 

    return {
        "statusCode": statuscode,
        "headers": headers,
        "body"  : json.dumps(body)
    }  



# Our debug main - We use this for local debug as this main function is not used by lambda function.
if __name__ == '__main__':
    # print(json.dumps(getposturl("test","positive")))
    # test code to generate upload urls for individual and timetoken url
    ################
    a=json.dumps(getindividualurl("60dce6999bf955be9f38b50d142cc3dc28c803c09162b51502c1dc072055dca7","demosite"))
    print(a)
    # print("----------------")
    # print(json.dumps(gettimetokenurl("unknown")))
    ################
    # print(generateindividualtokens(10))
    # print(generatetimetoken(10))
