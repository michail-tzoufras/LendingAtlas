import os
import csv
import argparse

import pandas as pd
import numpy as np
import copy

import matplotlib.pyplot as plt
plt.rc("font", size=14)
import matplotlib as mpl
mpl.rcParams['legend.frameon'] = 'True'


import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

#------FOR LOGISTIC REGRESSION AND RANDOM FOREST---------------------
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

#------FOR EMBEDDINGS---------------------
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Dense, Input
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model

#----------------------#----------------------#----------------------

def new_csv_writer( path, name, filenumber, headers, delimiter):
    """Returns a csv writer with the proper headers"""
    writer =  csv.writer( 
        open(
            os.path.join( path, name % filenumber ),'w'
            ), delimiter=delimiter
        )
    writer.writerow(headers)
    return writer

def split(filehandler, output_path, output_name_template, row_limit=100000, delimiter=','):
    """Break up the file provided by the filehandler to several csv files
    of manageable size of row_limit rows. """

    file_reader = csv.reader(filehandler, delimiter=delimiter)
    file_headers = next(file_reader)

    current_chunk = 1
    i = row_limit   # this initialization allows us to start a 
                    # new output_writer when entering the loop below 

    for row in file_reader:
        if ( (i+1) > row_limit ):
            output_writer = new_csv_writer(output_path, 
                                           output_name_template, 
                                           current_chunk, file_headers, delimiter)
            current_chunk += 1
            i = 0

        output_writer.writerow(row)
        i += 1

#----------------------#----------------------#----------------------

def data_exploration(df):
    """Preliminary data exploration function, plots things..."""

    print(df['Status'].value_counts())
    count_no_sub = len(df[df['Status']=='paid'])
    count_sub = len(df[df['Status']=='defaulted'])
    pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
    print("Paid = %.2f"%(pct_of_no_sub*100)+"%")
    pct_of_sub = count_sub/(count_no_sub+count_sub)
    print("Defaulted = %.2f"%(pct_of_sub*100)+"%")


    pd.crosstab(df['Sector'],df['Status']).plot(kind='bar',figsize=(15, 6),cmap='Set1')
    plt.title('Status Frequency per Sector')
    plt.xlabel('Sector')
    plt.ylabel('Status')
    plt.tight_layout(pad=1.0)
    plt.savefig('Status_Freq_per_Sector.png')

    table=pd.crosstab(df['Country'],df['Status'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(15, 6),cmap='Set1',legend=None)
    #pd.crosstab(df_r.Country,df_r.Status).plot(kind='bar', stacked=True,figsize=(15, 6),cmap='Set1')
    #plt.xlabel('Country')
    plt.ylabel('Status Probability')
    plt.subplots_adjust(bottom=0.45) # or whatever
    plt.savefig('Status_Probability_Per_Country.png')

def normalize_column(df_column, center_at_zero = False):
    """Converts an unnormalized dataframe column to a normalized 
    1D numpy array
    Default: normalizes between [0,1]
    (center_at_zero == True): normalizes between [-1,1] """

    normalized_array = np.array(df_column,dtype = 'float64')
    amax, amin = np.max(normalized_array), np.min(normalized_array)
    normalized_array -= amin
    if center_at_zero:
        normalized_array *= 2.0/(amax-amin)
        normalized_array -= 1.0
    else:
        normalized_array *= 1.0/(amax-amin)
    return normalized_array

#----------------------#----------------------#----------------------

def dataframe_to_numpy(df,categorical_columns,ordinal_columns):
    """Converts data in dataframe to numpy array that includes:
    1) one-hot encoded categorical columnhs
    2) normalized ordinal columns"""

    le = preprocessing.LabelEncoder()
    Xtmp = (df[categorical_columns].copy()).apply(lambda col: le.fit_transform(col))

    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore',sparse=False )
    X = np.transpose(ohe.fit_transform(Xtmp))

    for c in ordinal_columns:        
        X = np.vstack([X, normalize_column(df[c])])

    return np.transpose(X)

def sklearn_report(test,predictions):
    print('Log-Loss = %.2f'%(log_loss(test,predictions)))
    print("\nReport:")
    print(classification_report(test,predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(test,predictions))

#----------------------#----------------------#----------------------

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


    def __call__(self, _input_data, _labels, quiet = False):
        """compiles the model, fits the _input_data to the _labels, and evaluates the accuracy
        of the merged_model"""

        # compile the model
        (self.merged_model).compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        # fit the model
        (self.merged_model).fit(_input_data, _labels, epochs=2, verbose=0)

        # evaluate the model
        loss, accuracy = (self.merged_model).evaluate(_input_data, _labels, verbose=0)

        if (not quiet):
            plot_model(self.merged_model, to_file='merged_model_plot.png', 
                                        show_shapes=True, show_layer_names=True)

        return accuracy
        
def data_preprocessing(df, categorical_columns, vocabulary_sizes, max_length, ordinal_columns):
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

    # normalize the orindal features
    ordinal_data = [ (normalize_column(df[c])).reshape(-1,1)
                             for c in ordinal_columns] 

    # merge categorical and ordinal feature inputs
    return (padded_categorical_data+ordinal_data)



#-----------------------------------
#------------MAIN-------------------
#-----------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', 
                        default='/Users/mtzoufras/Desktop/Insight/Insight_Project_Data/BigML_Dataset.csv',
                        help='dataset path')
    parser.add_argument('--solver', action='store', 
                        default='Logistic Regression',
                        help="Select solver from: (1) 'Logistic Regression' \
                                                  (2) 'Random Forest' \
                                                  (3) 'Embeddings' \
                                                  (4) 'All' ")
    args = parser.parse_args()

    datapath = '/Users/mtzoufras/Desktop/Insight/Insight_Project_Code/d0.0_Make_CSVs/split3/'
    dataname = 'BigML_Split_%s.csv'
    #split(open(args.data, 'r'), datapath, dataname)

    df_raw = pd.read_csv(datapath+dataname%1)

    useful_columns = ['Loan Amount','Country','Sector','Activity','Status']
    valid_status = ['paid','defaulted']

    df_clean = (df_raw[useful_columns][df_raw.Status.isin(valid_status)]).copy()
    df_clean['Funded Time'] = ((df_raw['Funded Date.year']+0.0833*df_raw['Funded Date.month'])
                                [df_raw.Status.isin(valid_status)]).copy()

    data_exploration(df_clean)
    categorical_columns = ['Country','Sector','Activity']
    ordinal_columns = ['Loan Amount','Funded Time']

    if ((args.solver == 'Logistic Regression') or \
        (args.solver == 'Random Forest') or \
        (args.solver == 'All')):
        
        X  = dataframe_to_numpy(df_clean,categorical_columns,ordinal_columns)
        y = np.array((pd.get_dummies(df_clean['Status'], columns=['Status'])['paid']).tolist())


        # Split data set into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
        if ((args.solver == 'Logistic Regression') or (args.solver == 'All')):
            # Import module for fitting
            logmodel = LogisticRegression(solver='lbfgs',class_weight= {0:.9, 1:.1},
                              penalty='l2',C=10.0, max_iter=500)

            # Fit the model using the training data
            logmodel.fit(X_train, y_train)

            y_pred = logmodel.predict(X_test)
            print('\nLogistic Regression')
            print('-------------------\n')
            sklearn_report(y_test,y_pred)

        if ((args.solver == 'Random Forest') or (args.solver == 'All')):
            rfmodel = RandomForestClassifier(n_estimators=25,class_weight= {0:.9, 1:.1},max_depth=10)
            rfmodel.fit(X_train,y_train)
            probabilities = rfmodel.predict_proba(X_test)
            y_pred = np.array(probabilities[:,0] < probabilities[:,1]).astype(int)

            #print(classification_report(y_test,predictions))
            print('Random Forest')
            print('-------------------\n')
            sklearn_report(y_test,y_pred)


    if ((args.solver == 'Embeddings') or (args.solver == 'All')):
        
        # Find the vocabulary sizes for the categorical features
        vocabulary_sizes = [df_clean[c].nunique() for c in categorical_columns]

        # Maximum sentence length
        max_length=2

        embeddings_model = model_with_embeddings(vocabulary_sizes, max_length, len(ordinal_columns))

        input_data = data_preprocessing(df_clean, categorical_columns, vocabulary_sizes, max_length, ordinal_columns)
        labels = np.array((pd.get_dummies(df_clean['Status'], columns=['Status'])['paid']).tolist())

        acc = embeddings_model(input_data,labels)

        print('Accuracy: %f' % (acc*100))
