# Developed by: Michail Tzoufras 
# Date updated: 9/28/2019

import os
import numpy as np

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers import BatchNormalization, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model

#import visualization as Vis


class model_with_embeddings(object):
    """This is a class that implements a basic model with embeddings"""

    def __init__(self, vocabulary_sizes, max_length, _categorecal_features, num_ordinal_features, dense_nodes,
        pretrained_embeddings, quiet = False):
        """Setup the model based on the sizes of the feature arrays"""

        # Note that the vocabulary size will have to accomm
        nodes_in_embedding_layer = [max(2, int(np.ceil(np.sqrt(np.sqrt(v)))))
                                    for v in vocabulary_sizes]

        # Create embeddings for the categorical inputs
        embedding_inputs = []
        flat_embeddings = []
        models = []
        self.emb_names = [ (c.replace(" ", "_") +'_embedding') for c in _categorecal_features]

        for i,vocab_size in enumerate(vocabulary_sizes):

            embedding_inputs.append(Input(shape=(max_length,)))
            if (len(pretrained_embeddings) == 0):
                embedding_i = Embedding(vocab_size, nodes_in_embedding_layer[i], 
                                     name = self.emb_names[i],
                                     input_length=max_length,#weights=[word_weight_matrix], 
                                     trainable=True)(embedding_inputs[i]) 
            else:
                embedding_i = Embedding(vocab_size, nodes_in_embedding_layer[i], 
                                     name = self.emb_names[i],
                                     input_length=max_length,
                                     weights=[pretrained_embeddings[i]], 
                                     trainable=False)(embedding_inputs[i]) 



            flat_embeddings.append(Flatten()(embedding_i))
            models.append(Model(inputs=embedding_inputs[i], outputs=flat_embeddings[i]))
        
        # Merge embeddings with ordinal inputs
        ordinal_inputs = [Input(shape = (1, )) for i in range(num_ordinal_features)]
        concatenated = concatenate(flat_embeddings+ordinal_inputs)

        # Deep network after all inputs have been incorporated
        hidden_layers=[concatenated]
        for i in range(len(dense_nodes)):
            hidden_layer = Dense(dense_nodes[i], activation='relu')(
                BatchNormalization()(hidden_layers[i]))
            hidden_layers.append(hidden_layer)


        output = Dense(1, activation='sigmoid')(hidden_layers[-1])
        self.merged_model = Model(inputs=embedding_inputs+ordinal_inputs, outputs=output)

        #print(self.merged_model.summary())
        if (not quiet):
            if (len(pretrained_embeddings) == 0):
                plot_model(self.merged_model, to_file='train_embeddings.png', 
                                        show_shapes=True, show_layer_names=True)
            else:
                plot_model(self.merged_model, to_file='pretrained_embeddings.png', 
                                        show_shapes=True, show_layer_names=True)


    def train(self, train_input_data, train_labels, _epochs, _batch_size):
        """compiles the model, fits the _input_data to the _labels, and evaluates the accuracy
        of the merged_model"""

        # compile the model
        (self.merged_model).compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # fit the model
        history = (self.merged_model).fit(train_input_data, train_labels, batch_size=_batch_size, 
                                          epochs=_epochs, verbose=1)
        
        return history


    def test_accuracy(self, _input_data, _labels, quiet = False):
        """evaluate the accuracy of the model""" 
        test_loss, test_accuracy = (self.merged_model).evaluate(_input_data, _labels, verbose=0)

        return test_accuracy

    def predict_prob(self, _input_data ):
        """predict probabilities for test set"""
        yhat_probs = (self.merged_model).predict(_input_data, verbose=0)

        return yhat_probs[:, 0]

    def predict(self, _input_data ):
        """predict classes for test set"""
        yhat_probs = (self.merged_model).predict(_input_data, verbose=0)

        return (yhat_probs[:, 0] > 0.5).astype(int)

    def embeddings_names(self):
        return self.emb_names

    def extract_weights(self,name):
        """Extract weights from a neural network model"""
    
        # Extract weights
        weight_layer = (self.merged_model).get_layer(name)
        weights = weight_layer.get_weights()[0]

        # Normalize
        # weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
        return weights

#------------------------------------------------------------------------------
        
class One_Hot_Encoder(object):
    """A one-hot encoder-decoder for categorical features"""
    
    def __init__(self, df, categorical_columns):
        """Create the dictionaries corresponding to this dataframe and the
        categorical columns"""
        self.num_to_name = {}
        self.name_to_num = {}
        for c in categorical_columns:
            self.num_to_name[c] = dict(enumerate(df[c].unique()))
            self.name_to_num[c] = {v: k for k, v in self.num_to_name[c].items()}


    def encode(self, df, categorical_columns, max_length):
        """ create one-hot representations for the categorical features
        pad the representations to a desired length"""
        padded_categorical_data = []
        for c in categorical_columns:
            ohe = [ [self.name_to_num[c][i]] for i in df[c] ]

            padded_categorical_data.append(
                pad_sequences(ohe, maxlen=max_length, padding='post')
            )

        # merge categorical and ordinal feature inputs
        return padded_categorical_data
    
    def retrieve_names(self,categorical_column, num_list):
        """Converts a list of numbers to the corresponding
        categorical variable names. Returns the list of names"""
        return [self.num_to_name[categorical_column][i] for i in num_list]

#------------------------------------------------------------------------------

class embeddings_models(object):
    """A functor that returns the trained model given the relevant inputs"""

    def __init__(self, vocabulary_sizes, max_length, categorical_columns, num_ordinal_features, _visualize):
        """Initialize functor: 
        All models trained will contain the same properties listed below"""

        self.vocabulary_sizes = vocabulary_sizes
        self.max_length = max_length
        self.categorical_columns = categorical_columns
        self.num_ordinal_features = num_ordinal_features
        self.visualize = _visualize
        self.num_models = 1

    def __call__(self, _input_data_train, _labels_train, 
                       _input_data_test, _labels_test,
                        _epochs, _batch_size, _layers , _embs_list = []):
        """Train a model, print the test accuracy, and return the trained model"""

        model = model_with_embeddings(self.vocabulary_sizes, self.max_length, self.categorical_columns,
                                      self.num_ordinal_features, _layers, _embs_list)
        train_history = model.train(_input_data_train, _labels_train, _epochs, _batch_size)
        (self.visualize).plot_training_history(train_history,'train_history_'+str(self.num_models)+'.png')
        test_acc = model.test_accuracy(_input_data_test,_labels_test)
        print('Testing Accuracy = %f' % (test_acc*100))
        self.num_models+=1

        return model

