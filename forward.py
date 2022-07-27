import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import pytesseract
import cv2

outputMapping = {0: "education", 1: "jobb", 2: "skill", 3: "name",
                 4: "email", 5: "phonenumber"}

model = tf.keras.models.load_model('models/parser.model')

vectorizer = pickle.load(open('models/vectorizer.pickle', 'rb'))
selector = pickle.load(open("models/selector.pickle", "rb"))

img = cv2.imread('cv/CV2.png')
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
out_below = pytesseract.image_to_string(img)

for line in out_below.split("\n"):
    input = line

    plangs = []
    frameworks = []
    languages = []

    with open("datasets/programming.txt", 'r') as f:
        plangs = f.readlines()
    with open("datasets/frameworks.txt", 'r') as f:
        frameworks = f.readlines()
    with open("datasets/languages.txt", 'r') as f:
        languages = f.readlines()

    if (input.lower() + "\n") in plangs:
        print(line + " equals to plang")
    elif (input.lower() + "\n") in frameworks:
        print(line + " equals to framework")
    elif (input.lower()) in languages:
        print(line + " equals to language")
    else:
        X = vectorizer.transform([input])
        X = selector.transform(X).astype('float32')

        predictions = model.predict(X)
        if predictions[0][np.argmax(predictions)] > 0.9:
            print(line + " equals to " + outputMapping[np.argmax(predictions)])
            print("----------------")
