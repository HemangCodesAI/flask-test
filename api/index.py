from PIL import Image
import numpy as np
import pickle
from flask import Flask, request

app = Flask(__name__)
def predict(image):
    resized_image = image.resize((32, 32))
    img_array = np.array(resized_image.convert('RGB'))
    normalized_image = img_array / 255.0
    model = pickle.load(open("finalized_model.pkl", 'rb'))
    ans = model.predict(normalized_image.reshape(1, 32, 32, 3))
    print("huhua")
    if ans[0][0] < ans[0][1]:
        return "Non-Anemic"
    else:
        return "Anemic"

@app.route("/")
def index():
    return "chal raha hai"    

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        file = request.files['file']
        image = Image.open(file)
        prediction_result = predict(image)
        print({"prediction_result": prediction_result})
        return {"prediction_result": prediction_result}
    except Exception as e:
        return {"error": str(e)}

        
