#!/usr/bin/env python

from sklearn.metrics import roc_curve, auc, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np
import os

plt.switch_backend('agg')

def draw_ROC_PRC_integrated2(Yscore_Ytrue_ToolName, out_dir, Comparison_name, add_points=None, add_points_name=None, draw=True, colors_list=None, pos_class_ratio=None, ticks_size=15, savefig=True):  
    """
    Yscore_Ytrue_ToolName: a list of tuples consisting of the elements: y_score, y_true, software name
    Xy_dfs_cols_clfs_names: a list of tuples consisting of the elements: Xy_df, ycol, classifier, clf_name
    add_points: a tuple consisting of ("sensitivity", "1-specificity", "precision")
    pos_class_ratio: (# positive samples)/(# negative samples), used in drawing the random case of PR-AUC.
    """
    print("Start!")
    if colors_list is None:
        colors_list=['darkblue', 'darkgreen', 'coral', 'orchid', 'chocolate', 
                    'fuchsia', 'gold', 'yellowgreen', 'lightblue', 'green', 'indigo', 'maroon',
                    'lavender', 'lightgreen','navy', 'lime', 'magenta', 'orange', 
                    'pink', 'olive', 'purple']
    #random.shuffle(colors_list)
    if(draw==True):
            plt.clf()
            fig=plt.figure(figsize=(8.0, 16.0))
            ax1=fig.add_subplot(2,1,1)
            ax2=fig.add_subplot(2,1,2)
            
    
    # Compute ROC curve and area the curve
    roc_auc_list=[]
    i=0
    for y_scores, y_true, tool_name in Yscore_Ytrue_ToolName:
        print(tool_name)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        prc_auc = average_precision_score(y_true, y_scores)
        if(draw==True):
            ax1.plot(fpr, tpr, lw=2, color=colors_list[i], label=tool_name+' (area = %0.3f)' % (roc_auc))
            ax2.plot(rec, prec, lw=2, color=colors_list[i], label=tool_name+' (area = %0.3f)' % (prc_auc))        
        i+=1
        roc_auc_list.append(roc_auc)
        
    if(draw==True):
        if not (add_points is None):
            ax1.plot(add_points[1], add_points[0], '.', markersize=12, label= add_points_name+' (SE, 1-SP) = (%0.3f, %0.3f)' % (add_points[0], add_points[1]))
            ax2.plot(add_points[0], add_points[2], '.', markersize=12, label= add_points_name+' (SE, PR) = (%0.3f, %0.3f)' % (add_points[0], add_points[2]))

        ax1.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random'+' (area = 0.500)')
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize='x-large')
        ax1.set_ylabel('True Positive Rate', fontsize='x-large')
        ax1.set_title('ROC-AUC', fontsize='xx-large')
        ax1.legend(loc="lower right", fontsize='x-large')
        #ax1.set_axis_bgcolor('white')
        if pos_class_ratio:
            ax2.plot([0, 1], [pos_class_ratio, pos_class_ratio], '--', color=(0.6, 0.6, 0.6), label='Random'+' (area = %0.3f)' % (pos_class_ratio))
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize='x-large')
        ax2.set_ylabel('Precision', fontsize='x-large')
        ax2.set_title('PR-AUC', fontsize='xx-large')
        ax2.legend(loc="upper right", fontsize='large')
        #ax2.set_axis_bgcolor('white')
        for ax in [ax1,ax2]:
            for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(ticks_size)
            for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(ticks_size)
        if savefig:
            fig.savefig(os.path.join(out_dir,Comparison_name+'_ROC_PR.pdf'), format='pdf')
            
        #fig.show()
    return np.array(roc_auc_list)