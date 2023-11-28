from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('model/final_model.h5')  # Update this path to where your model is stored

@app.route('/', methods=['GET'])
def home():
    return render_template('upload.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'prediction': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # Resize and preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    
    # Convert prediction to label
    label = 'Hotdog' if prediction[0][0] < 0.5 else 'Not hotdog'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)



