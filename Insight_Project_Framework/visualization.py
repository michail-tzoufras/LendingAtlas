import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rc("font", size=14)
import matplotlib as mpl
mpl.rcParams['legend.frameon'] = 'True'

import seaborn as sns
sns.set(style="white")

#---------------------------------------------------------------------
from sklearn import metrics 
from mpl_toolkits.axes_grid1 import make_axes_locatable


def country_vs_status(df):
    Country_names = []
    Number_Defaults = []
    Number_Paid = []
    for i,r in enumerate(df['Country'].unique()):
        Country_names.append(r)
        ftmp = df[df['Country']==r]['Status']
        if ((ftmp.value_counts()).shape[0] < 2):
            defaults = ftmp[ftmp == 'defaulted'].value_counts()
            if (defaults.shape[0] > 0):
                Number_Defaults.append(np.sum(ftmp.value_counts()))
                Number_Paid.append(0)
            else:
                Number_Defaults.append(0)
                Number_Paid.append(np.sum(ftmp.value_counts()))
        else:
            defaults = ftmp[ftmp == 'defaulted'].value_counts()
            Number_Defaults.append(defaults[0])
            Number_Paid.append(np.sum(ftmp.value_counts())-defaults[0])

    Percentage = np.array([d*100.0/float(p+d) for p,d in zip(Number_Paid,Number_Defaults)])

    indices = np.argsort(np.array(Percentage))
    Countries = np.array(Country_names)[indices]
    Defaults = np.array(Number_Defaults)[indices]
    Paid = np.array(Number_Paid)[indices]

    max_digits = int(np.log10(np.max(Paid+Defaults)))
    yticks = [1]
    for i in range(max_digits+1):
        yticks.append(yticks[i]*10)
    yticklabels=[str(ytick) for ytick in yticks]

    fig = plt.figure(figsize=(20, 9))
    ax = fig.add_subplot(1, 1, 1)
    plt.title('Countries listed in order of default rate',fontsize = 16)
    plt.ylabel('Number of Loans',fontsize = 16)
    plt.bar(range(Countries.shape[0]), Defaults, color='red',label = 'defaulted')
    plt.bar(range(Countries.shape[0]), Paid,  bottom=Defaults, color='black',label = 'paid')
    plt.xticks(range(Countries.shape[0]), Countries, rotation='vertical',fontsize = 12)
    for ytick in yticks[:-1]:
        plt.axhline(y=ytick, color='white', linestyle='dotted')

    plt.subplots_adjust(bottom=0.35) # or whatever

    ax.set_ylim([0.8,yticks[-1]])
    ax.set_yscale('log')
    ax.set_xlim([-0.5,Countries.shape[0]+0.5])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=16)
    plt.legend(fontsize = 16)
    plt.savefig('Countries_vs_status.png')


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

    #table=pd.crosstab(df['Country'],df['Status'])
    #table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(15, 6),cmap='Set1',legend=None)
    #plt.ylabel('Status Probability')
    #plt.subplots_adjust(bottom=0.45) # or whatever
    #plt.savefig('Status_Probability_Per_Country.png')



#----------------------#----------------------#----------------------

def plot_confusion_matrix(ytest, yhat_list, model_title_list):
    """Plots the confusion matrix for a list of models
    with the corresponding list of predicionts yhat given
    the ground truth ytest."""

    MAX_VALUE = 10000

    for yhat, model_title in zip(yhat_list, model_title_list):
        
        # Get the confusion matrix
        cm = metrics.confusion_matrix(ytest,yhat) 
        
        # Initialize the figure
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        im = ax.matshow(cm,cmap=plt.cm.Blues,vmin=0, vmax=MAX_VALUE)
        plt.title(model_title,y=-0.15,fontsize = 18, fontweight='bold')

        # Resize the colorbar to fit the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=14)

        # Add the labels
        ax.set_xticklabels([''] + ['paid','defaulted'],fontsize=16)
        ax.set_yticklabels([''] + ['paid','defaulted'],fontsize=16)
    
        # Add the text inside the confusion matrix
        x_start, y_start = 0.0, 0.0
        x_end,   y_end  = 2.0, 2.0
        size=2
        jump_x = (x_end - x_start) / (2.0 * size)
        jump_y = (y_end - y_start) / (2.0 * size)
        x_positions = np.linspace(start=x_start, stop=x_end, num=size, endpoint=False)
        y_positions = np.linspace(start=y_start, stop=y_end, num=size, endpoint=False)
        x_positions -= 0.5
        y_positions -= 0.5
        
        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = cm[y_index,x_index]
                text_x = x + jump_x
                text_y = y + jump_y
                number_color = 'black'
                if (label > 0.7*MAX_VALUE):
                    number_color = 'white'         
                ax.text(text_x, text_y, label, color=number_color, ha='center', va='center',fontsize=16)
    
        fig.tight_layout(pad=2)
        fig.savefig(model_title + ' Confusion Matrix.png')


def plot_roc_curve(ytest, yprob_list, model_title_list):
    """Plots the ROC curve for a list of models
    with the corresponding list of probabilities yprob given
    the ground truth ytest. All curves are shown on the same
    plot"""
    
    Colors = ['DodgerBlue','ForestGreen']

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    #plt.plot([0, 1], [0, 1], linestyle='--', color = 'Black' )
    for i, model_title in enumerate(model_title_list):
        fpr, tpr, thresholds = metrics.roc_curve(ytest,yprob_list[i])
        plt.plot(fpr, tpr, marker='.', color = Colors[i], label = model_title)

    # Add the labels
    ax.set_ylabel('True Positive Rate',fontsize=16)
    ax.set_xlabel('False Positive Rate',fontsize=16)
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
    plt.legend(fontsize = 16)

    fig.savefig('ROC_Curve.png')

def plot_precision_recall(ytest, yhat_list, yprob_list, model_title_list):
    """Plots the precision-recall for a list of models
    with the corresponding list of probabilities yprob given
    the ground truth ytest. All curves are shown on the same
    plot."""
    
    Colors = ['DodgerBlue','ForestGreen']

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    #random_rate = np.sum(y_test)/float(y_test.shape[0])
    #plt.plot([0, 1], [random_rate, random_rate], linestyle='--',color = 'black')
    for i, model_title in enumerate(model_title_list):
        precision, recall, thresholds = metrics.precision_recall_curve(ytest,yprob_list[i])
        auc = metrics.auc(recall, precision)
        f1 = metrics.f1_score(ytest, yhat_list[i])
        ap = metrics.average_precision_score(ytest, yprob_list[i])
        #print(model_title+':  f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
        plt.plot(recall, precision, marker='.',color=Colors[i], label = model_title + ", AUC =%.3f"%auc)
        
    ax.set_ylabel('Precision',fontsize=16)
    ax.set_xlabel('Recall',fontsize=16)
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
    plt.legend(fontsize = 16)
    fig.savefig('Precision Recall.png')

        
def report_model_performance(ytest, yhat_list, yprob_list, model_title_list):
    """Calls the relevant functions to plot model performance:
    1) the confusion matrix for each model
    2) the ROC curve for all models
    3) the precision-recall curve for all models"""
    
    plot_confusion_matrix(ytest, yhat_list, model_title_list)
    plot_roc_curve(ytest, yprob_list, model_title_list)
    plot_precision_recall(ytest, yhat_list, yprob_list, model_title_list)
