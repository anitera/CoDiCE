import os
import requests


def download(dataset_name):
    if dataset_name == "homeloan":
        download_homeloan_dataset()
    elif dataset_name == "diabetes":
        download_diabetes_dataset()
    elif dataset_name == "income":
        download_income_dataset()
    else:
        raise ValueError("Dataset is not supported for downloading. You may download it manually and place it in the datasets folder")
    

def download_homeloan_dataset():
    # https://docs.google.com/spreadsheets/d/1iOABOgrcNKVKbmSlU10k2kuSUYAnDs1q/view?usp=sharing
    file_id = "1iOABOgrcNKVKbmSlU10k2kuSUYAnDs1q"  # replace with your file id
    # If datasets folder is not created, create it
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
    destination = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'homeloan_train.xls')
    download_from_google_drive(file_id, destination)

def download_diabetes_dataset():
    # https://drive.google.com/file/d/1DRcP6jK9zW0ZYshIDqZq6x3DKRJnkv-e/view?usp=sharing
    file_id = "1DRcP6jK9zW0ZYshIDqZq6x3DKRJnkv-e"  # replace with your file id
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
    destination = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'diabetes_train.xls')
    download_from_google_drive(file_id, destination)

def download_income_dataset():
    # https://drive.google.com/file/d/1wYmn1SXQ2cNttwjQd3GK7JqiLdURTofy/view?usp=sharing
    file_id = "1wYmn1SXQ2cNttwjQd3GK7JqiLdURTofy"  # replace with your file id
    if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')):
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
    destination = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'income_train.xls')
    download_from_google_drive(file_id, destination)


def download_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params = {'id': file_id}, stream = True)
    
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
            
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params = params, stream = True)
        
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)