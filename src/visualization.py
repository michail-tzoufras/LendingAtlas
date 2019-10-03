# Developed by: Michail Tzoufras 
# Date updated: 10/2/2019

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

def rank_by_status(df, feature):
    Feature_names = []
    Number_Defaults = []
    Number_Paid = []
    for i,r in enumerate(df[feature].unique()):
        Feature_names.append(r)
        ftmp = df[df[feature]==r]['Status']
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
    Names = np.array(Feature_names)[indices]
    Defaults = np.array(Number_Defaults)[indices]
    Paid = np.array(Number_Paid)[indices]

    return Names, Defaults, Paid


def make_tiers(df, c, number_tiers):
    """Split the data into tiers. First tier has the 
    highest likelihood to exhibit class 1."""

    # Rank the data in this feature so that you can split
    # into tiers
    ranked_feature, Defaults, Paid = rank_by_status(df,c)
            
    # First tier includes all the data with no Defaults
    i = 0
    tiers = []
    tier = []
    while (Defaults[i] == 0):
        tier.append(ranked_feature[i])
        i += 1
    if (len(tier)>0):
        tiers.append(tier)

    # Last tier includes all the data with all Defaults
    # Make it now and append it later
    last = len(Defaults)-1
    last_tier = []
    while (Paid[last] == Paid[-1]):
        last_tier.append(ranked_feature[last])
        last -= 1

    # Find the length of the remaining tiers
    tier_length = int((last - i + 1)/(
        number_tiers-int(len(last_tier)>0)-int(len(tier)>0)
            ))

    # Create the remaining tiers; the next-to-last tier may have a few more
    # elements to make up for the integer division
    while (i < last+1):
        tier = []
        this_tier = 0
        # Fill in the rest of the tiers except the next-to-last tier
        if (len(tiers) < (number_tiers - int(len(last_tier)>0) - 1 ) ):
            while (this_tier < tier_length):
                tier.append(ranked_feature[i])
                i+=1
                this_tier+=1
        # Put the remaining elements in the next-to-last tier
        else:
            while (i < last+1):
                tier.append(ranked_feature[i])
                i+=1
        tiers.append(tier)

    # It is time to append the last tier 
    if (len(last_tier)>0):
        tiers.append(last_tier)

    return tiers


def find_similar(name, output_path, weights,  index, rindex, n = 10):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results
    modified from: https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526"""
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
            
    # Find furthest and closest items
    furthest = sorted_dists[:(n // 2)]
    closest = sorted_dists[-n-1: len(dists) - 1]
    items = [rindex[c] for c in furthest]
    items.extend(rindex[c] for c in closest)
        
    # Find furthest and closets distances
    distances = [dists[c] for c in furthest]
    distances.extend(dists[c] for c in closest)
        
    colors = ['tomato' for _ in range(n //2)]
    colors.extend('mediumspringgreen' for _ in range(n))
        
    data = pd.DataFrame({'distance': distances}, index = items)
        
    # Horizontal bar chart
    fig = plt.figure(figsize=(9, 9))

    data['distance'].plot.barh(color = colors, alpha = 0.9, figsize = (10, 8),
                                   edgecolor = 'k', linewidth = 2)
    plt.xlabel('Cosine Similarity',fontsize = 16);
    plt.axvline(x = 0, color = 'k')
    plt.tight_layout()
    #plt.subplots_adjust(left=0.5, right=0.5)
    plt.tick_params(labelsize=12)
        
    # Formatting for italicized title
    name_str =  'Most and Least Similar to' #f'{index_name.capitalize()}s
    for word in (str(name)).split():
        # Title uses latex for italize
        name_str += ' $\it{' + word + '}$'
    plt.title(name_str, x = 0.2, size = 12, y = 1.2)
    plt.tight_layout()
    fig.savefig(output_path +'cosine_similarity_'+word+'.png')
        
    return None


class Make_Visualizations(object):

    def __init__(self,output_path):
        self.path= output_path 

    def country_vs_status(self, df):
        Countries, Defaults, Paid = rank_by_status(df,'Country')

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
        plt.savefig(self.path+'Countries_vs_status.png')


    def data_exploration(self, df):
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
        plt.savefig(self.path+'Status_Freq_per_Sector.png')

    #table=pd.crosstab(df['Country'],df['Status'])
    #table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=(15, 6),cmap='Set1',legend=None)
    #plt.ylabel('Status Probability')
    #plt.subplots_adjust(bottom=0.45) # or whatever
    #plt.savefig('Status_Probability_Per_Country.png')


#----------------------#----------------------#----------------------
    def report_numerical(self, ytest, yhat_list, model_title_list):
        """Reports Precision Recall and f1 score for all the models"""
    
        for i, model_title in enumerate(model_title_list):
            precision = metrics.precision_score(ytest, yhat_list[i],average='binary')
            recall = metrics.recall_score(ytest, yhat_list[i],average='binary')
            f1 = metrics.f1_score(ytest, yhat_list[i])
            print("--------------------")
            print(model_title)
            print("--------------------")
            print('Precision = %.3f'%precision)
            print('Recall = %.3f'%recall)
            print('f1 score = %.3f'%f1)



    def plot_confusion_matrix(self, ytest, yhat_list, model_title_list):
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
            fig.savefig(self.path+model_title + ' Confusion Matrix.png')


    def plot_roc_curve(self, ytest, yprob_list, model_title_list):
        """Plots the ROC curve for a list of models
        with the corresponding list of probabilities yprob given
        the ground truth ytest. All curves are shown on the same
        plot"""
    
        Colors = ['DodgerBlue','ForestGreen','Crimson','Black']

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

        fig.savefig(self.path+'ROC_Curve.png')

    def plot_precision_recall(self, ytest, yhat_list, yprob_list, model_title_list):
        """Plots the precision-recall for a list of models
        with the corresponding list of probabilities yprob given
        the ground truth ytest. All curves are shown on the same
        plot."""
    
        Colors = ['DodgerBlue','ForestGreen','Crimson','Black']

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        #random_rate = np.sum(y_test)/float(y_test.shape[0])
        #plt.plot([0, 1], [random_rate, random_rate], linestyle='--',color = 'black')
        for i, model_title in enumerate(model_title_list):
            precision, recall, thresholds = metrics.precision_recall_curve(ytest,yprob_list[i])
            auc = metrics.auc(recall, precision)
            prec = metrics.precision_score(ytest, yhat_list[i],average='binary')
            rec = metrics.recall_score(ytest, yhat_list[i],average='binary')
            f1 = metrics.f1_score(ytest, yhat_list[i])
            ap = metrics.average_precision_score(ytest, yprob_list[i])
            #print(model_title+':  f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
            plt.plot(recall, precision, marker='.',
                color=Colors[i], label = model_title + ", AUC =%.3f"%auc)
        
        ax.set_ylabel('Precision',fontsize=16)
        ax.set_xlabel('Recall',fontsize=16)
        ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=16)
        plt.legend(fontsize = 12)
        fig.savefig(self.path+'Precision Recall.png')

        
    def report_model_performance(self, ytest, yhat_list, yprob_list, model_title_list):
        """Calls the relevant functions to plot model performance:
        1) --> reports the precision, recall and f1 score
        2) the confusion matrix for each model
        3) the ROC curve for all models
        4) the precision-recall curve for all models"""

        self.report_numerical(ytest, yhat_list, model_title_list)    
        self.plot_confusion_matrix(ytest, yhat_list, model_title_list)
        self.plot_roc_curve(ytest, yprob_list, model_title_list)
        self.plot_precision_recall(ytest, yhat_list, yprob_list, model_title_list)


    def plot_training_history(self, _hist,_filename):
        """Plotting the training history"""
        fig = plt.figure(figsize=(8,8))
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(_hist.history['loss'], label='train')
        #plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        #plot accuracy during training
        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(_hist.history['acc'], label='train')
        #plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.subplots_adjust(hspace=0.5)
        fig.savefig(self.path+_filename)



class Visualize_Embeddings(object):

    def __init__(self, embeddings_path, output_path,
            df, clms):

        self.status_colors = ['forestgreen','limegreen','lime','gold','orange','salmon','red']
        self.embeddings_path = embeddings_path
        self.output_path= output_path 
        self.feature_tiers = {}
        for c in clms:
            self.feature_tiers[c] = make_tiers(df,c,len(self.status_colors))

    def reduce_dim(self,_weights, components = 3):
        """Reduce dimensions of embeddings"""
        from sklearn.manifold import TSNE
        return TSNE(components, metric = 'cosine').fit_transform(_weights)

    def weights_n_labels(self, filename):
        df = pd.read_csv(filename)
        cols = df.columns
        labels = df[cols[-1]]
        weights = df[cols[:-1]].to_numpy()
        if (weights.shape[1] > 2):
            reduced_weights = self.reduce_dim(weights, components = 2)
            return reduced_weights, labels
        else:
            return weights, labels

    def plot_embeddings(self, feature, filename, output_filename, interesting_values, tiers=[]):
        """ Make a 2D visualzation of the embeddings.
        Highlight the interesting values in the list below."""

        weights,labels = self.weights_n_labels(self.embeddings_path+filename)
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(1, 1, 1)
        plt.xlabel('TSNE 1',fontsize = 16)
        plt.ylabel('TSNE 2', fontsize=16)
        plt.title(feature, fontsize = 18)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)


        color_ = 'black'
        for j, tier in enumerate(tiers):
            xaxis = []
            yaxis = []
            for i,txt in enumerate(labels):
                if (txt in tier):
                    if (txt in interesting_values):                
                        ax1.annotate(txt, (weights[i,0], weights[i,1]),weight='bold',
                                   fontsize = 16)
                        ax1.scatter(weights[i,0], weights[i,1],alpha = 1.0, 
                                    c = self.status_colors[j], marker = '*', s=250,edgecolor = 'b')
                    else: 
                        xaxis.append(weights[i,0])
                        yaxis.append(weights[i,1])


            ax1.scatter(xaxis,yaxis,alpha = 0.5, c = self.status_colors[j], s=150)

        fig1.savefig(self.output_path+output_filename)


    def plot_embeddings_similarity(self, feature, value):
        """Calls 'find similar' to plot the cosine similarity for this value of this feature"""
        weights,labels = self.weights_n_labels(self.embeddings_path+feature+'_embedding.csv')
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        weights= weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
        find_similar(value, self.output_path, weights, label_to_index, index_to_label)


    def display(self,feature, interesting_values =[]):
        self.plot_embeddings(feature, feature+'_embedding.csv',feature+'_embedding_plot.png',
            interesting_values, self.feature_tiers[feature])
