from flask import Flask, render_template, request
import json
import numpy as np
from tensorflow.python.keras.models import load_model
import pickle
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Load model dengan objek kustom SGD
from keras.optimizers import SGD


# Load model, tokenizer, stemmer, words, dan classes
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Mendaftarkan objek kustom SGD
custom_objects = {'SGD': SGD}

# Memuat model dengan compile=False dan menyertakan custom_objects
model = load_model('laundry.h5', compile=False, custom_objects=custom_objects)

tokenizer = RegexpTokenizer(r'\w+')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# Function untuk membersihkan kalimat
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Function untuk membuat bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function untuk memprediksi kelas intent
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function untuk mendapatkan respon berdasarkan intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Maaf, saya tidak mengerti apa yang Anda maksud."
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "Maaf, saya tidak mengerti apa yang Anda maksud."

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk menerima dan mengirim pesan
@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    ints = predict_class(user_text)
    res = get_response(ints, intents)
    return res

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
