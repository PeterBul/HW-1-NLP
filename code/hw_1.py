import tensorflow as tf
import keras as K
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import io
from os import path
from keras.layers import Bidirectional, LSTM

PAD = "<PAD>"
START = "<START>"
UNK = "<UNK>"
MAX_SIZE = 50
HIDDEN_SIZE = 256


def build_vocab(data):
    MAX_VOCAB_SIZE = 20000
    word_to_id = dict()
    word_to_id['<PAD>'] = 0
    word_to_id['<START>'] = 1
    word_to_id['<UNK>'] = 2
    index = 3
    if isinstance(data, io.TextIOWrapper):
        for line in data:
            line = line.strip()
            for i in range(len(line)):
                if(len(word_to_id) >= MAX_VOCAB_SIZE):
                    id_to_word = {v: k for k, v in word_to_id.items()}
                    return word_to_id, id_to_word
                if line[i] not in word_to_id:
                    word_to_id[line[i]] = index
                    index += 1
                if i < len(line) - 2:
                    if line[i:i + 2] not in word_to_id:
                        word_to_id[line[i:i + 2]] = index
                        index += 1
    elif isinstance(data, str):
        for i in range(len(data)):
            if (len(word_to_id) >= MAX_VOCAB_SIZE):
                id_to_word = {v: k for k, v in word_to_id.items()}
                return word_to_id, id_to_word
            if data[i] not in word_to_id:
                word_to_id[data[i]] = index
                index += 1
            if i < len(data) - 2:
                if data[i:i + 2] not in word_to_id:
                    word_to_id[data[i:i + 2]] = index
                    index += 1

    id_to_word = {v: k for k, v in word_to_id.items()}

    return word_to_id, id_to_word


with open(
        "/Users/petercbu/Google Drive/University/2019 Spring/NLP/hw1_nlp_sapienza_2019/resources/train/input/as.txt") as f:
    word_to_id_as, id_to_word_as = build_vocab(f)


VOCAB_SIZE = len(word_to_id_as)


def create_input_dataset(file, word_to_id):
    x = []
    for line in file:
        feature_vector = []
        feature_vector.append(word_to_id[START])

        # Build feature vector
        for i in range(len(line)):
            unigram = line[i]
            if unigram in word_to_id:
                feature_vector.append(word_to_id[unigram])
            else:
                feature_vector.append(word_to_id[UNK])

            if i < len(line) - 2:
                bigram = line[i:i + 2]
                if bigram in word_to_id:
                    feature_vector.append(word_to_id[bigram])
                else:
                    feature_vector.append(word_to_id[UNK])

        x.append(np.array(feature_vector))
    return np.array(x)


def BIES_to_numerical(file_path):
    BIES_to_number = {'B': 0, 'I': 1, 'E': 2, 'S': 3}
    y = []
    with open(file_path, 'r', encoding='utf-8') as f:

        for line in f:
            line = line.strip()
            new_line = []
            for ch in line:
                new_line.append(str(BIES_to_number[ch]))
            y.append(new_line)
    return np.array(y)


with open(
        "/Users/petercbu/Google Drive/University/2019 Spring/NLP/hw1_nlp_sapienza_2019/resources/train/input/as.txt") as f:
    train_x_as = create_input_dataset(f, word_to_id_as)

with open(
        "/Users/petercbu/Google Drive/University/2019 Spring/NLP/hw1_nlp_sapienza_2019/resources/dev/input/as.txt") as f:
    dev_x_as = create_input_dataset(f, word_to_id_as)

train_y_as = BIES_to_numerical(
    "/Users/petercbu/Google Drive/University/2019 Spring/NLP/hw1_nlp_sapienza_2019/resources/train/labels/as.txt")
dev_y_as = BIES_to_numerical(
    "/Users/petercbu/Google Drive/University/2019 Spring/NLP/hw1_nlp_sapienza_2019/resources/dev/labels/as.txt")

train_x_as = pad_sequences(train_x_as, truncating='pre', padding='post', maxlen=MAX_SIZE)
dev_x_as = pad_sequences(dev_x_as, truncating='pre', padding='post', maxlen=MAX_SIZE)
train_y_as = pad_sequences(train_y_as, truncating='pre', padding='post', maxlen=MAX_SIZE)
dev_y_as = pad_sequences(dev_y_as, truncating='pre', padding='post', maxlen=MAX_SIZE)

# print(train_x_as[0:2])
# print(train_y_as[0:2])
# print(dev_x_as[0:2])
# print(dev_y_as[0:2])

train_y_as = K.utils.to_categorical(train_y_as, 4, dtype='int')
dev_y_as = K.utils.to_categorical(dev_y_as, 4, dtype='int')

print(train_y_as.shape)
print(dev_y_as.shape)


def create_keras_model(vocab_size, embedding_size, hidden_size, dropout, recurrent_dropout):
    print("Creating KERAS model")

    # define LSTM
    print("Vocab size: " + str(vocab_size))
    print("Embedding size: " + str(embedding_size))
    model = K.models.Sequential()
    model.add(K.layers.Embedding(vocab_size, embedding_size, mask_zero=True))
    model.add(Bidirectional(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True)))
    model.add(K.layers.TimeDistributed(K.layers.Dense(4, activation='softmax')))

    # we are going to use the Adam optimizer which is a really powerful optimizer.
    optimizer = K.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

batch_size = 32
epochs = 3
EMBEDDING_SIZE = 150
model = create_keras_model(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, 0.2, 0.2)
# Let's print a summary of the model
model.summary()

cbk = K.callbacks.TensorBoard("logging/keras_model")
print("\nStarting training...")
model.fit(train_x_as, train_y_as, epochs=epochs, batch_size=batch_size,
          shuffle=True, validation_data=(dev_x_as, dev_y_as), callbacks=[cbk])
print("Training complete.\n")

#print("\nEvaluating test...")
#loss_acc = model.evaluate(test_x, test_y, verbose=0)
#print("Test data: loss = %0.6f  accuracy = %0.2f%% " % (loss_acc[0], loss_acc[1]*100))