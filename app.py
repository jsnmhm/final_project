from flask import Flask, request, jsonify
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
model = load_model('final_model.h5')  # Update this path to where your model is stored

@app.route('/', methods=['GET'])
def home():
    return '''
    <html>
        <body>
            <h1>Image Classifier</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" onchange="previewImage(event)">
                <img id="imagePreview" style="max-width: 500px; max-height: 500px; display: none;"/>
                <button type="button" onclick="classifyImage()">Classify</button>
            </form>
            <div id="predictionResult"></div>

            <script>
                function previewImage(event) {
                    var reader = new FileReader();
                    reader.onload = function() {
                        var output = document.getElementById('imagePreview');
                        output.src = reader.result;
                        output.style.display = 'block';
                    };
                    reader.readAsDataURL(event.target.files[0]);
                }

                function classifyImage() {
                    var formData = new FormData(document.getElementById('uploadForm'));
                    fetch('/classify', {
                        method: 'POST',
                        body: formData
                    }).then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    }).then(data => {
                        document.getElementById('predictionResult').innerText = 'Prediction: ' + data.prediction;
                    }).catch(error => {
                        console.error('There has been a problem with your fetch operation:', error);
                    });
                    event.preventDefault();
                }
            </script>
        </body>
    </html>
    '''

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
    label = 'hotdog' if prediction[0][0] < 0.5 else 'nothotdog'

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)


