# encoding:utf-8


import requests

def predict_result(input_seq):

    PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict/'
    payload = {"message":input_seq}
    # Submit the request.
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()

    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        print(r['result'])
    # Otherwise, the request failed.
    else:
        print('Request failed')
    print(r)

predict_result("你好")