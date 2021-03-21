import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import json
import random
import pickle
import time
import webbrowser
import flask
import os
import difflib
import nltk
from flask import Flask, render_template, request

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
from tensorflow.python.framework import ops
import json
import pickle
import random
import time
import difflib
import numpy
import webbrowser
import tflearn
import tensorflow
import random
import flask
import os


app = Flask(__name__)
with open("intents.json") as file:
    data = json.load(file)

# try:
#     with open("data.pickle", "rb") as f:
#         words,labels,training,output = pickle.load(f)
# except:
words = []
labels = []
docs_X = []
docs_Y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds =nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_X.append(wrds)
        docs_Y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words =[stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels =sorted(labels)

training =[]
output =[]

out_empty =[0 for _ in range(len(labels))]
for x,doc in enumerate(docs_X):
    bag=[]

    wrds =[stemmer.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row=out_empty[:]
    output_row[labels.index(docs_Y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)
## till here
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#     model.load("model.tflearn")
# except:
model.fit(training,output,n_epoch =1400,batch_size=8, show_metric=True)
model.save("model.tflearn")
def bag_of_words(s,words):
    bag =[0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words =[stemmer.stem(words.lower()) for words in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat(inp):
##    query = ask_the_bot()
##    blank = ask_the_bot()
    #print("BOT FLEX is online to answer")
    while True:
        if inp.lower() == "quit":
            return "Good Bye"
            break


        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return random.choice(responses)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    time.sleep(1)
    return str(chat(userText))

if __name__ == "__main__":
    print("*"*60)
    time.sleep(1)
    print("*"*40)
    time.sleep(1)
    print("*"*20)
    time.sleep(1)
    print("ChatBot will be launched in your web browser in..")
    time.sleep(1)
    print("3 sec")
    time.sleep(1)
    print("2 sec")
    time.sleep(1)
    print("1 sec")
    time.sleep(1)
    print("0 sec")
    webbrowser.open("http://127.0.0.1:5000/")
    app.run()





