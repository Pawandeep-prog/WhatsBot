from keras.models import load_model
import pickle 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
import numpy as np

model = load_model('chatbot.h5')

LENGTH=8

with open("dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)
inv_dict = {v: k for k, v in dictionary.items()}

query=""
while query!="q":
    query = input("YOU: ")
    try:
        query = [[dictionary[word] for word in query.split(" ")]]
        query = pad_sequences(query, maxlen=LENGTH, padding="post", truncating="post")
        bot_init=["<sos>"]
        op_length=0
        idx = 0
        word=""
        while op_length<LENGTH and word!="<eos>":
            bot_enc = [[dictionary[i] for i in bot_init]]
            in_seq = pad_sequences(bot_enc, truncating="post", padding="post", maxlen=LENGTH)
            op = model.predict([query, in_seq])
            word = inv_dict[np.argmax(op[0][idx][:])]
            op_length = op_length+1
            bot_init.append(word)
            print(word)
            idx=idx+1

    except:
        print("ERR")

    print("----------")
print("BYE")


