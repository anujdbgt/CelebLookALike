#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, render_template, jsonify
import face_recognition
import os
import numpy as np
from IPython.display import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CELEBRITY_FOLDER'] = 'celebrity_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

known_encodings = []
known_images = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def load_images():
    global known_encodings, known_images
    known_images_dir = app.config['CELEBRITY_FOLDER']

    for file in os.listdir(known_images_dir):
        if allowed_file(file):
            filename = os.fsdecode(file)
            image = face_recognition.load_image_file(os.path.join(known_images_dir, filename))
            enc = face_recognition.face_encodings(image)
            
            if len(enc) > 0:
                known_encodings.append(enc[0])
                known_images.append(filename)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = face_recognition.load_image_file(filepath)
            image_to_test_encoding = face_recognition.face_encodings(image)[0]
            face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)

            top_indices = np.argsort(face_distances)[:3]

            matches = []
            for index in top_indices:
                match = {
                    'name': known_images[index],
                    'similarity': float(1-face_distances[index])*100
                }
                matches.append(match)
            return jsonify({'matches': matches})
    
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            
    return jsonify({'error': 'Invalid file format'}), 400                                         

load_images()

if __name__ == '__main__':
    app.run(debug=True)



