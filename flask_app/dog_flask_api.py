# TODO build flask app backend
from flask import Flask, render_template, Response, request, send_file
from dog_breed_classifier import DogBreedClassifier
import os
app = Flask(__name__)
dog_breed_classifier = DogBreedClassifier()


@app.route('/')
def index():
    temp = "D:/ai_nd/DogBreedClassifier/test_images/bella/bella.jpg"
    return render_template('index.html', user_image=temp)


@app.route('/handle_request', methods=['POST'])
def handle_request():
    image_path = request.form['text']
    # return render_template('index.html', user_image=image_path)
    return render_template('index.html', user_image=image_path)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)  # host='0.0.0.0',
