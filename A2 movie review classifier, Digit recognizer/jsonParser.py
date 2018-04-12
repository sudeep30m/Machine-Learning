import json
import argparse
import urllib
import requests

def download(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


argparser = argparse.ArgumentParser(
    prog='JSON parser',
    description='Parsing download links in JSON')
argparser.add_argument('params', type = int, nargs = 2, help = 'questionNumber, modelNumber')
# argparser.add_argument('parameter', type = float, nargs = 1, help = 'c')
args = argparser.parse_args()
question = str(args.params[0])
model =  str(args.params[1])
links = json.load(open('links.json'))
links = links[question][model]
# testfile = urllib.URLopener()
path = str(question) + "/" + str(model) + "/"
if(int(question) == 2 and int(model) > 1):
    # print("yeah")
    path = path +'libsvm/python/'
for key in links: 
    download(links[key], path + key)
