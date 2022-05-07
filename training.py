from calendar import EPOCH
from gc import callbacks
import numpy as np 
from keras.layers import Dense, Input, Embedding, LSTM
from keras.models import Model
from keras.optimizers import rmsprop_v2
import pickle
import sys
import matplotlib.pyplot as plt

from keras.callbacks import Callback

class InferCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%25==0:
            print("="*30)
            print(epoch)

            model.save("model.h5")

            queries = ["hi","thanks","how are you","hello","kal jana hai", "hor ki krde", "chlo vadia", "kal class hogi", "college jana hai"]

            for query in queries:
                print(query, ":  ")
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
                print("----------")


EPOCHS = int(sys.argv[1])

LENGTH=8
#from tensorflow.keras.utils import plot_model

from keras.preprocessing.sequence import pad_sequences

question = np.load('questions.npy')
answers = np.load('answers.npy')
answers_orig = np.load('answers_orig.npy')
with open("dictionary.pkl", "rb") as f:
    dictionary = pickle.load(f)

inv_dict={v:k for k,v in dictionary.items()}

print("# loading files done 👍")
print("# creating model ⏳")

neuron=400

enc_inp = Input(shape=(LENGTH, ), name="enc_inp")
dec_inp = Input(shape=(LENGTH, ), name="dec_inp")

VOCAB_SIZE = len(dictionary)
embed = Embedding(VOCAB_SIZE+1, output_dim=400, 
                  input_length=LENGTH,
                  trainable=True,
                  mask_zero=True                  
                  )

enc_embed = embed(enc_inp)

# enc_lstm_2=LSTM(200, return_sequences=True)(enc_embed)

enc_op, h, c = LSTM(neuron, return_sequences=True, return_state=True)(enc_embed)

enc_states = [h, c]

dec_embed = embed(dec_inp)

dec_lstm_l = LSTM(neuron, return_sequences=True, return_state=False)

dec_lstm = dec_lstm_l(dec_embed, initial_state=enc_states)

# dec_op=LSTM(units=400, return_sequences=True, return_state=False)(dec_lstm)

# dense_middle_layer = Dense(512, activation="tanh")(dec_op)

# dense_middle_layer = Dense(1024, activation="tanh")(dec_op)

dense_op = Dense(VOCAB_SIZE+1, activation='softmax')(dec_lstm)

model = Model([enc_inp, dec_inp], dense_op)

model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer=rmsprop_v2.RMSprop())

print(model.summary())

history = model.fit([question, answers_orig],answers,epochs=EPOCHS, callbacks=[InferCallback()])
# plt.plot(history.history['acc'])
# plt.plot(history.history['loss'])
# plt.show()
model.save("chatbot.h5")
