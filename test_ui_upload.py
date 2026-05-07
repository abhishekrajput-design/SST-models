import requests
import sys

url = "http://127.0.0.1:8080/api/upload?filename=20260505T073055769_385036.mp3&model=parakeet-tdt-0.6b-v3"
file_path = "C:/Users/abhis/Desktop/SST-models/testing-audio/omar_test/20260505T073055769_385036.mp3"

with open(file_path, 'rb') as f:
    data = f.read()

print("Uploading file to local UI...")
resp = requests.post(url, data=data)
print("Status:", resp.status_code)
print("Response:", resp.text)
