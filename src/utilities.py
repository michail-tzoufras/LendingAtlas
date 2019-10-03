# Developed by: Michail Tzoufras 
# Date updated: 9/29/2019

import os
import csv

import pandas as pd
import numpy as np
import copy

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

def reader(filehandler,  delimiter=','):
    """Break up the file provided by the filehandler to several csv files
    of manageable size of row_limit rows. """

    file_reader = csv.reader(filehandler, delimiter=delimiter)
    file_headers = next(file_reader)
        
    loan_id = []
    partner_id = []
    
    for i,row in enumerate(file_reader):
        if (i > 5):
            loan_id.append(row[0])
            partner_id.append(row[19])
    return loan_id, partner_id


def attach_partner_id(file1 = '/Users/mtzoufras/Desktop/Insight/Insight_Project_Data/kiva_ds_csv/loans.csv',
                      file2 = '/Users/mtzoufras/Desktop/Insight/Insight_Project_Data/BigML_Dataset.csv'):

    loan_ids, partners_ids = reader(open(file1,'r'))
    df_raw = pd.read_csv(file2)
    df_raw['Partner ID'] = 'Missing'
    list_of_ints = [np.int64(l) for l in loan_ids]

    for i,lid in enumerate(list_of_ints):
        if len(df_raw[df_raw['id'] == lid])>0:
            ind = df_raw.loc[df_raw['id'] == list_of_ints[i]].index.values[0]
            df_raw.at[ind,'Partner ID'] = partners_ids[i]

    useful_columns = ['Loan Amount','Country','Sector','Activity','Status','Funded Date.year','Funded Date.month','Partner ID']
    valid_status = ['paid','defaulted']
    df_clean = (df_raw[useful_columns][df_raw.Status.isin(valid_status)]).copy()
    df_clean.to_csv('dfclean.csv',mode = 'w', index=False)


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

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue