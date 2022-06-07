import face_recognition
import numpy as np
import urllib.request
import json
from flask import Flask, request, abort
import logging

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

TOLERANCE = 0.59
MODEL = 'cnn'  # 'hog' or 'cnn' - CUDA accelerated (if available) deep-learning pretrained model


@app.route("/FaceRecognitionTraining", methods=['POST'])
def Training_Faces():
    if not request.json or 'urls' not in request.json:
        abort(400)

    all_urls = json.loads(request.json['urls'])

    known_names = []
    known_faces = []

    for key, value in all_urls.items():
        response = urllib.request.urlopen(value)
        image = face_recognition.load_image_file(response)

        encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(encoding)
        known_names.append(key)

    data = {}
    for key in known_names:
        for value in known_faces:
            data[key] = value.tolist()
            known_faces.remove(value)
            break

    new_data = json.dumps(data)
    result_dict = {"output": new_data}
    return result_dict


@app.route("/FaceRecognitionTesting", methods=['POST'])
def Recognize_Face():
    if not request.json or 'encodings' not in request.json:
        abort(400)

    if not request.json or 'url' not in request.json:
        abort(400)

    all_face_encodings = json.loads(request.json['encodings'])

    img_url = request.json['url']

    known_names = list(all_face_encodings.keys())
    known_faces = np.array(list(all_face_encodings.values()))

    response = urllib.request.urlopen(img_url)
    image = face_recognition.load_image_file(response)

    locations = face_recognition.face_locations(image, model=MODEL)

    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            result = "I Found " + match

        check = all(element == False for element in results)

        if check:
            result = "I Found Unknown Person"

    result_dict = {"output": result}
    return result_dict


# app.run(debug=True)
