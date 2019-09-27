import os
import numpy as np

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model


class model_with_embeddings(object):
    """This is a class that implements a basic model with embeddings"""

    def __init__(self, vocabulary_sizes, max_length, num_ordinal_features):
        """Setup the model based on the sizes of the feature arrays"""

        # Note that the vocabulary size will have to accomm
        nodes_in_embedding_layer = [max(2, int(np.ceil(np.sqrt(np.sqrt(v))))) 
                                    for v in vocabulary_sizes]

        # Create embeddings for the categorical inputs
        embedding_inputs = []
        flat_embeddings = []
        models = []

        for i,vocab_size in enumerate(vocabulary_sizes):
            embedding_inputs.append(Input(shape=(max_length,)))
            embedding_i = Embedding(vocab_size+1, nodes_in_embedding_layer[i], input_length=max_length,#weights=[word_weight_matrix], 
                                trainable=True)(embedding_inputs[i]) 

            flat_embeddings.append(Flatten()(embedding_i))
            models.append(Model(inputs=embedding_inputs[i], outputs=flat_embeddings[i]))
        
        # Merge embeddings with ordinal inputs
        ordinal_inputs = [Input(shape = (1, )) for i in range(num_ordinal_features)]
        concatenated = concatenate(flat_embeddings+ordinal_inputs)

        # Deep network after all inputs have been incorporated
        hidden_1 = Dense(3, activation='relu')(concatenated)
        output = Dense(1, activation='sigmoid')(hidden_1)

        self.merged_model = Model(inputs=embedding_inputs+ordinal_inputs, outputs=output)


    def train(self, _input_data, _labels, quiet = False):
        """compiles the model, fits the _input_data to the _labels, and evaluates the accuracy
        of the merged_model"""

        # compile the model
        (self.merged_model).compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # fit the model
        (self.merged_model).fit(_input_data, _labels, epochs=10, verbose=0)

        # evaluate the model
        loss, accuracy = (self.merged_model).evaluate(_input_data, _labels, verbose=0)

        if (not quiet):
            plot_model(self.merged_model, to_file='merged_model_plot.png', 
                                        show_shapes=True, show_layer_names=True)

        return accuracy


    def test(self, _input_data, _labels, quiet = False):

        # evaluate the model
        loss, accuracy = (self.merged_model).evaluate(_input_data, _labels, verbose=0)

        return accuracy
        
def data_preprocessing(df, categorical_columns, vocabulary_sizes, max_length):
    """This is the provisinal pre-processing function, 
    it will need to be modified significantly when I include multiple files etc."""

    # create one-hot representations for the categorical features
    # pad the representations to a desired length
    padded_categorical_data = []
    for c,v in zip(categorical_columns, vocabulary_sizes):
        onehot_representations = [one_hot(column, v) for column in df[c]]
        padded_categorical_data.append(
            pad_sequences(onehot_representations, maxlen=max_length, padding='post')
        )

    # merge categorical and ordinal feature inputs
    return padded_categorical_data