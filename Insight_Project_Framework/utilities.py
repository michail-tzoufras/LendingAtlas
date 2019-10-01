# Developed by: Michail Tzoufras 
# Date updated: 9/23/2019

import os
import csv

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

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue