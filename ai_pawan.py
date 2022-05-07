from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle

data = np.array(open("chat.txt").read().split('\n'))

LENGTH=8

chat_inter = []
for i in data:
    if "]" in i:
        chat_inter.append(i.split("]")[-1])
    else:
        chat_inter[len(chat_inter)-1] = chat_inter[len(chat_inter)-1]+" "+i

chat_combined = [chat_inter[0]]
for i in range(1, len(chat_inter)):
    if chat_inter[i-1].split(":")[0] == chat_inter[i].split(":")[0]:
        chat_combined[len(chat_combined)-1] = chat_combined[len(chat_combined)-1] + chat_inter[i].split(":")[-1]
    else:
        chat_combined.append(chat_inter[i])

questions=[]
answers=[]
for i in chat_combined:
    answers.append(i.split(":")[-1]) if i.split(":")[0]==" Pawandeep" else questions.append(i.split(":")[-1])

for idx in range(len(questions)):
    questions[idx] = questions[idx].lower()[1:]

for idx in range(len(answers)):
    answers[idx] = answers[idx].lower()[1:]


#print(questions)
#ommiting last message from question 
print(len(questions), len(answers))
questions = questions[:len(answers)]

#adding <sos> and <eos>
for idx in range(len(answers)):
    answers[idx] = "<sos> "+answers[idx]+" <eos>"


for idx in range(len(questions)):
    questions[idx] = questions[idx].split(" ")[:LENGTH]
for idx in range(len(answers)):
    answers[idx] = answers[idx].split(" ")[:LENGTH]


for i in range(len(answers)):
    print(questions[i] , " -- ", answers[i])

#rint(questions)
#dictionary
dictionary = {}
id=1
for i in questions:
    for j in i:
        if j not in dictionary:
            dictionary[j]=id 
            id = id+1
for i in answers:
    for j in i:
        if j not in dictionary:
            dictionary[j]=id 
            id = id+1
dictionary["<pad>"] = 0


encoded_questions = []
encoded_answers = []
max_len=-1

for i in questions:
    encoded_questions.append([dictionary[word] for word in i])
    max_len = len(i) if len(i) > max_len else max_len
for i in answers:
    encoded_answers.append([dictionary[word] for word in i])
    max_len = len(i) if len(i) > max_len else max_len

#print(questions)
#print("============\n")
#print(answers)
print("len of questions & answers ==> ", len(encoded_questions), len(encoded_answers))
print("max length ==> ", max_len)
print("dictionary size ==> ", len(dictionary))

encoded_questions = pad_sequences(encoded_questions, maxlen=LENGTH, padding="post", truncating="post")
encoded_answers = pad_sequences(encoded_answers, maxlen=LENGTH, padding="post", truncating="post")

encoded_answers_op = []
for i in encoded_answers:
    encoded_answers_op.append(i[1:])
encoded_answers_op = pad_sequences(encoded_answers_op, maxlen=LENGTH, padding="post", truncating="post")

#print(dictionary)
encoded_answers_final = to_categorical(encoded_answers_op, num_classes=len(dictionary)+1)

print("shape of encoded questions & answers ==> ", encoded_questions.shape, encoded_answers.shape, encoded_questions.dtype, encoded_answers.dtype)
print("shape of final encoded answers ==> ", encoded_answers_final.shape)

np.save("questions.npy", encoded_questions)
np.save("answers.npy", encoded_answers_final)
np.save("answers_orig.npy", encoded_answers)

#print(encoded_questions[0][1])
#print(encoded_answers[0])
#print(encoded_answers_final[0])

with open("dictionary.pkl", "wb") as f:
    pickle.dump(dictionary, f)