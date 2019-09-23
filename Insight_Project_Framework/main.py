# Developed by: Michail Tzoufras 
# Date updated: 9/23/2019

import os
import csv
import argparse

import pandas as pd
import numpy as np
import copy
#------------------------------------------------------------------

import matplotlib.pyplot as plt
plt.rc("font", size=14)
import matplotlib as mpl
mpl.rcParams['legend.frameon'] = 'True'

import seaborn as sns
sns.set(style="white")
#------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#------------------------------------------------------------------

import visualization as Vis
import data_processing as Process

#------------------------------------------------------------------

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



def str2bool(v):
    """Allows me to use default bools in argparse"""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-----------------------------------
#------------MAIN-------------------
#-----------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', 
                        default='/Users/mtzoufras/Desktop/Insight/Insight_Project_Data/BigML_Dataset.csv',
                        help='dataset path')
    parser.add_argument('--solver', action='store', 
                        default='All',
                        help="Select solver from: (1) 'All' \
                                                  (2) 'Random Forest' \
                                                  (3) 'Embeddings' \
                                                  (4) 'Logistic Regression' ")
    parser.add_argument('--sample', action='store',
                        default='undersample',
                        help="For imbalanced classes: (1) 'undersample' \
                                                      (2) 'oversample' \
                                                      (3) 'None' ")
    parser.add_argument('--merge_rare', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Merge rare values")
    parser.add_argument('--explore_data', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Generate data exploration plots")


    args = parser.parse_args()

    #datapath = '/Users/mtzoufras/Desktop/Insight/Insight_Project_Code/d0.0_Make_CSVs/split3/'
    datapath = os.getcwd()+'/../data/preprocessed/'
    dataname = 'BigML_Split_%s.csv'
    #split(open(args.data, 'r'), datapath, dataname)

    df_raw = pd.read_csv(datapath+dataname%1)
    #df_raw = pd.read_csv('/Users/mtzoufras/Desktop/Insight/Insight_Project_Data/BigML_Dataset.csv')

    useful_columns = ['Loan Amount','Country','Sector','Activity','Status']
    valid_status = ['paid','defaulted']

    df_clean = (df_raw[useful_columns][df_raw.Status.isin(valid_status)]).copy()
    df_clean['Funded Time'] = ((df_raw['Funded Date.year']+0.0833*df_raw['Funded Date.month'])
                                [df_raw.Status.isin(valid_status)]).copy()

    categorical_columns = ['Country','Sector','Activity']
    ordinal_columns = ['Loan Amount','Funded Time']

    if args.merge_rare:
        for c in categorical_columns:
            Process.combine_rare(df_clean,c)

    if args.explore_data:
        Vis.data_exploration(df_clean)
        Vis.country_vs_status(df_clean)

    # Split data set into training and test sets
    df_train_raw, df_test = train_test_split(df_clean, test_size=0.3 )

    # Undersample the majority (default) or ...
    if (args.sample == 'undersample'):
        df_train = Process.undersample_majority(df_train_raw,2)
    # ... oversample the minority or ...
    elif (args.sample == 'oversample'):
        df_train = Process.oversample_minority(df_train_raw,0.5)
    # ... work with imbalance as is
    else:
        df_train = df_train_raw

    # Concatenate the train and test dataframes
    # to make sure the arrays have the same number
    # of features when they are split.
    train_length = len(df_train)
    df_concatenated = pd.concat([df_train,df_test])

    #print("Training set size = " + str(train_length))
    #print("Test set size = " + str(len(df_concatenated)-train_length))
    #print("Total set size = " + str(len(df_concatenated)))


    # Convert dataframe to numpy array and labels
    X_concatenated  = Process.dataframe_to_numpy(df_concatenated,categorical_columns,ordinal_columns)
    y_concatenated = np.array((pd.get_dummies(df_concatenated['Status'], columns=['Status'])['defaulted']).tolist())

    # Split the train from the test arrays
    X_train, X_test = X_concatenated[:train_length, :], X_concatenated[train_length:, :]
    y_train, y_test = y_concatenated[:train_length   ], y_concatenated[train_length:   ]


    y_pred, y_prob, model_titles = [], [], []
    if ((args.solver == 'Logistic Regression') or \
        (args.solver == 'Random Forest') or \
        (args.solver == 'All')):
        

        if ((args.solver == 'Logistic Regression') or (args.solver == 'All')):
            # Import module for fitting
            logmodel = LogisticRegression(solver='lbfgs',#class_weight= {0:.1, 1:.9},
                              penalty='l2',C=10.0, max_iter=500)

            # Fit the model using the training data
            logmodel.fit(X_train, y_train)

            y_pred.append(logmodel.predict(X_test))
            y_prob.append(logmodel.predict_proba(X_test)[:,1])
            model_titles.append('Logistic Regression')

        if ((args.solver == 'Random Forest') or (args.solver == 'All')):
            # Import module for fitting
            rfmodel = RandomForestClassifier(n_estimators=25,#class_weight= {0:.1, 1:.9},
                max_depth=10)

            # Fit the model using the training data
            rfmodel.fit(X_train,y_train)

            y_pred.append(rfmodel.predict(X_test))
            y_prob.append(rfmodel.predict_proba(X_test)[:,1])
            model_titles.append('Random Forest')


    if ((args.solver == 'Embeddings') or (args.solver == 'All')):

        import embeddings_DL as Emb
        
        # Find the vocabulary sizes for the categorical features
        vocabulary_sizes = [df_concatenated[c].nunique() for c in categorical_columns]

        # Maximum sentence length
        max_length=2

        embeddings_model = Emb.model_with_embeddings(vocabulary_sizes, max_length, len(ordinal_columns))
        categorical_data = Emb.data_preprocessing(df_concatenated, categorical_columns, vocabulary_sizes, max_length)

        # normalize the orindal features
        ordinal_data = [ (Process.normalize_column(df_concatenated[c])).reshape(-1,1) for c in ordinal_columns] 

        input_data = categorical_data+ordinal_data
        labels = np.array((pd.get_dummies(df_concatenated['Status'], columns=['Status'])['paid']).tolist())

        labels_train = labels[:train_length]
        labels_test = labels[train_length:]
        input_data_train = []
        input_data_test = []
        for i in range(len(input_data)):
            input_data_train.append(input_data[i][:train_length,:])
            input_data_test.append(input_data[i][train_length:,:])


        acc = embeddings_model.train(input_data_train,labels_train)
        print('Training Accuracy = %f' % (acc*100))
        acc = embeddings_model.test(input_data_test,labels_test)

        print('Testing Accuracy = %f' % (acc*100))


    Vis.report_model_performance(y_test,y_pred,y_prob,model_titles)