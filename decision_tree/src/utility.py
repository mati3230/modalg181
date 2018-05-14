'''
Created on 16.01.2018

@author: mati3230
'''

import numpy as np

def entropy(t):
    """
    Computes the impurity of the target set.

    Parameters
    ----------
    t : np.ndarray 
		column vector of the target feature you want to classify
    Returns
    -------
    decimal
        Entropy.

    """
    entrop = 0
	# TODO compute entropy here
    hist = {}
    num_rows = np.size(t, 0)
    for i in range(0, np.size(t,0)):
        if not t[i] in hist.keys():
            hist[t[i]] = 0
        hist[t[i]] += 1
        
    entrop = 0
    for val in hist.values():
        arg = val/num_rows
        entrop-=arg*np.log2(arg)
	
    return entrop

def partition(D, question):
    """
    Split the data into two data sets according to the answer of a certain question.

    Parameters
    ----------
    D : numpy.ndarray
        Dataset
    question : function
        Question of the feature. 
        
    Returns
    -------
    (np.ndarray, np.ndarray)
        Rows, where question is true and false
    """
    true_rows = []
    false_rows = []
    num_rows = np.size(D, 0)
    for i in range(0, num_rows):
        row = D[i,:]
		# TODO ask question and split to get true_rows and false_rows
		# TODO make use true_rows.append(row)
        if question(D=row):
            true_rows.append(row)
        else:
            false_rows.append(row)
		
    return (np.array(true_rows), np.array(false_rows))

def information_gain(D, question, t_col):
    """
    Computes information gain between a feature s and its values according to a question.
    
    Parameters
    ----------
    D : numpy.ndarray
        Dataset of current node
    question : function
        Question of a feature. 
    t_col : int
        Column number of feature, where you want to compute the info gain.
    
    Returns
    -------
    decimal
        Information Gain
    
    """
    # compute entropy of this node
    t = D[:,np.size(D, 1)-1] if t_col == None else D[:, t_col]
    num_rows = np.size(t,0)
    entrop = entropy(t=t)
    
    # make a split 
    (true_rows, false_rows) = partition(D=D, question=question)
    num_true_rows = np.size(true_rows,0)
    num_false_rows = np.size(false_rows,0)
    
    # compute child entropies
    entrop_true = 0 if num_true_rows == 0 else entropy(t=true_rows[:,np.size(D, 1)-1] if t_col == None else true_rows[:, t_col])
    
    entrop_false = 0 if num_false_rows == 0 else entropy(t=false_rows[:,np.size(D, 1)-1] if t_col == None else false_rows[:, t_col])
    
    # compute information gain
    information_gain = 0
    true_weight = num_true_rows / num_rows
    false_weight = num_false_rows / num_rows
    
    # TODO implement computation of information gain here
    # remember: entropy(parent) - [weighted average] entropy(children)
    
    information_gain = entrop - (true_weight * entrop_true + false_weight * entrop_false)
    
    return information_gain