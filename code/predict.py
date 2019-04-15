from argparse import ArgumentParser
import keras as K
from keras.layers import Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import io

PAD = "<PAD>"
UNK = "<UNK>"
MAX_SIZE = 24
HIDDEN_SIZE = 256


# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument("input_path", help="The path of the input file")
#     parser.add_argument("output_path", help="The path of the output file")
#     parser.add_argument("resources_path", help="The path of the resources needed to load your model")
#
#     return parser.parse_args()


def predict(x_uni, x_bi, y):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    VOCAB_SIZE_UNI = 5973
    VOCAB_SIZE_BI = 695107
    HIDDEN_SIZE = 256
    batch_size = 32
    epochs = 3
    EMBEDDING_SIZE = 32
    model = create_keras_model(VOCAB_SIZE_UNI, VOCAB_SIZE_BI, EMBEDDING_SIZE, HIDDEN_SIZE, 0.2, 0.2)
    model.load_weights('my_model_weights.h5')

    prediction = model.predict([x_uni, x_bi])

    number_to_bies = ['B', 'I', 'E', 'S']

    BIES_prediction = np.array([[number_to_bies[np.argmax(ch)] for ch in line] for line in prediction])

    with open("prediction.txt", 'a') as f:
        for line in BIES_prediction:
            f.write("".join(line))
            f.write("\n")



def create_keras_model(vocab_size_uni, vocab_size_bi, embedding_size, hidden_size, dropout, recurrent_dropout):
    print("Creating KERAS model")

    # define LSTM
    uni_input_layer = K.layers.Input((MAX_SIZE,))
    bi_input_layer = K.layers.Input((MAX_SIZE,))
    uni_embedding = K.layers.Embedding(vocab_size_uni, embedding_size, mask_zero=True)(uni_input_layer)
    bi_embedding = K.layers.Embedding(vocab_size_bi, embedding_size, mask_zero=True)(bi_input_layer)
    concatenated_layer = K.layers.concatenate([uni_embedding, bi_embedding], axis=-1)
    bidirectional_layer = Bidirectional(LSTM(hidden_size, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))(concatenated_layer)
    output = K.layers.TimeDistributed(K.layers.Dense(4, activation='softmax'))(bidirectional_layer)

    model = K.models.Model(inputs=[uni_input_layer, bi_input_layer], outputs=[output])

    # we are going to use the Adam optimizer which is a really powerful optimizer.
    optimizer = K.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def create_input_dataset(file, word_to_id_uni, word_to_id_bi):
    x_uni = []
    x_bi = []
    for line in file:
        line = line.strip()
        feature_vector_uni = []
        feature_vector_bi = []
        # Build feature vector
        for i in range(len(line)):
            unigram = line[i]
            if unigram in word_to_id_uni:
                feature_vector_uni.append(word_to_id_uni[unigram])
            else:
                feature_vector_uni.append(word_to_id_uni[UNK])

            if i < len(line) - 2:
                bigram = line[i:i + 2]
                if bigram in word_to_id_bi:
                    feature_vector_bi.append(word_to_id_bi[bigram])
                else:
                    feature_vector_bi.append(word_to_id_bi[UNK])

        x_uni.append(np.array(feature_vector_uni))
        x_bi.append(np.array(feature_vector_bi))
    return np.array(x_uni), np.array(x_bi)


def build_vocab(data):
    word_to_id_uni = dict()
    word_to_id_bi = dict()
    word_to_id_uni['<PAD>'] = 0
    word_to_id_uni['<UNK>'] = 1
    word_to_id_bi['<PAD>'] = 0
    word_to_id_bi['<UNK>'] = 1

    uni_index = 2
    bi_index = 2

    if isinstance(data, io.TextIOWrapper):
        for line in data:
            line = line.strip()
            for i in range(len(line)):
                if line[i] not in word_to_id_uni:
                    word_to_id_uni[line[i]] = uni_index
                    uni_index += 1
                if i < len(line) - 2:
                    if line[i:i + 2] not in word_to_id_bi:
                        word_to_id_bi[line[i:i + 2]] = bi_index
                        bi_index += 1

    id_to_word_uni = {v: k for k, v in word_to_id_uni.items()}
    id_to_word_bi = {v: k for k, v in word_to_id_bi.items()}

    return word_to_id_uni, id_to_word_uni, word_to_id_bi, id_to_word_bi

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


if __name__ == '__main__':
    with open(
            "../resources/train/input/as.utf8", 'r', encoding='utf-8') as f:
        word_to_id_as_uni, id_to_word_as_uni, word_to_id_as_bi, id_to_word_as_bi = build_vocab(f)
    VOCAB_SIZE_UNI = len(word_to_id_as_uni)
    VOCAB_SIZE_BI = len(word_to_id_as_bi)
    with open(
            "../resources/dev/input/as.utf8", 'r', encoding='utf8') as f:
        test_x_as_uni, test_x_as_bi = create_input_dataset(f, word_to_id_as_uni, word_to_id_as_bi)


    test_y_as = BIES_to_numerical(
        "../resources/dev/labels/as.utf8")

    test_x_as_uni = pad_sequences(test_x_as_uni, truncating='pre', padding='post', maxlen=MAX_SIZE)
    test_x_as_bi = pad_sequences(test_x_as_bi, truncating='pre', padding='post', maxlen=MAX_SIZE)
    test_y_as = pad_sequences(test_y_as, truncating='pre', padding='post', maxlen=MAX_SIZE)

    test_y_as = K.utils.to_categorical(test_y_as, 4, dtype='int')


    predict(test_x_as_uni, test_x_as_bi, test_y_as)
