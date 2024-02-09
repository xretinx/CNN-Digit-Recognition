from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
model2 = tf.keras.models.load_model('cnnModel')
# model3 = tf.keras.models.load_model('githubCopy')
# model4 = tf.keras.models.load_model('pretrainedGithubMane')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the drawn image from the request
        drawn_image = request.form['image']

        # Process the image
        processed_image = process_image(drawn_image)

        # Make a prediction using the loaded model
        prediction = model2.predict(processed_image)
        print(prediction)
        predicted_digit2 = np.argmax(prediction)
        # Return the predicted digit
        # return jsonify([int(predicted_digit2), int(predicted_digit3), int(predicted_digit4)])
        return jsonify({"prediction": str(predicted_digit2), "confidence": str(np.max(prediction))})

    return render_template('index.html')


def process_image(drawn_image):
    # Decode the base64 string
    image_data = base64.b64decode(drawn_image.split(',')[1])

    # Convert the image data to a numerical array
    img = Image.open(io.BytesIO(image_data)).convert('L')

    # Resize the image to 28x28 pixels
    img = img.resize((28, 28))

    # Convert the image to a numerical array
    img_array = np.array(img)

    # Normalize and reshape the image
    img = img_array.astype('float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    np.save('img', img)
    return img


if __name__ == '__main__':
    app.run()
