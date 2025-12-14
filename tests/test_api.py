import requests
url='http://127.0.0.1:8000/predict'

payload={
    "features":['0.9758786559455050','0.49976548004593600','0.017236932369880600','0.10490945933603600','0.1495648919013710','4']
}
try:
    response= requests.post(url, json=payload)

    if response.status_code==200:
          print("Prediction Response:", response.json())
    else:
        print(f"Error: {response.status_code} - {response.text}")

except Exception as e:
    print(f"An error occurred: {e}")