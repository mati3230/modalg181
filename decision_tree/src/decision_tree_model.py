'''
Created on 18.01.2018

@author: mati3230
'''
from utility import information_gain, partition
import numpy as np

class decision_tree_model(object):
    
    def __init__(self):
        self.depth_to_node = {}
        self.depth = 0
    
    def train(self, D, questions, t_col):
        """
        Calls the build_tree recursion function and saves the tree.
    
        Parameters
        ----------
        D : numpy.array
            Dataset
        questions : list
            List of questions for each feature in the data set
        Returns
        -------
        Tree
            Links between the decision and leaf nodes
    
        """
        self.depth = 0
        self.tree = self.build_tree(D=D, questions=questions, depth=0, t_col=t_col)
        print("training finished, depth: {0}".format(self.depth))
    
    def build_tree(self, D, questions, depth, t_col):
        """
        Recursion function of the decision tree. It figures out the decision nodes and leaf nodes.
    
        Parameters
        ----------
        D : numpy.array
            Dataset
        questions : list
            List of questions for each feature in the data set
        Returns
        -------
        Decision Node or Leaf Node
            Node of the tree.
    
        """
        # index of which question should be asked in order to split the tree
        split_col = 0 
        max_gain = 0
        question=questions[0]
        for i in range(0, len(questions)):
            # calculate information gain for a certain feature
            gain = information_gain(D=D, question=questions[i], t_col=t_col)
			# TODO determine column where you want to make split
			# TODO assign that column to split_col variable
			
			
        # end of the tree reached
        if max_gain == 0:        
            return leaf_node(D=D, depth=depth, t_col=t_col)
        # analyze true and false examples
        self.depth += 1
		# grow tree in depth
        (true_rows, false_rows) = partition(D=D, question=questions[split_col])
        true_branch = self.build_tree(D=true_rows, questions=questions, depth=depth+1, t_col=t_col)
        false_branch = self.build_tree(D=false_rows, questions=questions, depth=depth+1, t_col=t_col)
        return decision_node(question=questions[split_col], true_branch=true_branch, false_branch=false_branch, depth=depth, D=D, t_col=t_col)
        
    def test_tree(self, D, t_col):
        """
        Classifies a sample according to the target feature
    
        Parameters
        ----------
        D : numpy.array
            Vector, which represents a data sample
		t_col : int
			Column number of feature, where you want to compute the info gain.
        Returns
        -------
        class label
            Class label of the target feature.
    
        """
        num_rows = 1
        num_true_answers = 0
        num_false_answers = 0
        if np.ndim(D) == 2:
            num_rows=np.size(D,0)
        if num_rows == 1: # classify one example
            answer = self.tree.ask_question(D=D)
            if answer == D[t_col]:
                num_true_answers += 1
            else:
                num_false_answers += 1
        else: # classify multiple examples
            for i in range(0, num_rows): # iterate over examples
                ##################
                if type(self.tree) is leaf_node:
                    return self.tree.get_answer()
                else:
                    answer = self.tree.ask_question(D=D[i,:])
                ground_truth = D[i,t_col]
                if answer == ground_truth:
                    num_true_answers += 1
                else:
                    num_false_answers += 1
        accuracy = num_true_answers/num_rows
        return accuracy
    
    def classify(self, D):
        """
        Classifies a sample according to the target feature
    
        Parameters
        ----------
        D : numpy.array
            Vector, which represents a data sample
        Returns
        -------
        class label
            Class label of the target feature.
    
        """
        num_rows = 1
        if np.ndim(D) == 2:
            num_rows=np.size(D,1)
        if num_rows == 1:
            return np.ndarray(self.tree.ask_question(D=D))
        else:
            answers = []
            for i in range(0, num_rows):
                answers.append(self.tree.ask_question(D=D[i,:]))
        return np.ndarray(answers)
        
class leaf_node(object):
    
    def __init__(self, D, depth, t_col):
		# return answer of class with highest evidence
        self.answer = np.argmax(np.bincount(D[:, t_col].astype(dtype=np.int64)))
        self.depth=depth
    
    def get_answer(self):
        return self.answer
    
    def get_depth(self):
        return self.depth
    
class decision_node(object):
    
    def __init__(self, question, true_branch, false_branch, depth, D, t_col):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth=depth
		# return answer of class with highest evidence
        self.answer = np.argmax(np.bincount(D[:, t_col].astype(dtype=np.int64)))
    
    def get_node_answer(self):
        return self.answer
    
    def get_answer(self):
        return self.answer
    
    def ask_question(self, D):
        """
        Recursion of asking questions till an answer is found.
    
        Parameters
        ----------
        D : numpy.array
            Dataset
        Returns
        -------
        class label
            Class of the target feature.
    
        """
        
		# TODO iterate through the tree till you get an answer
		# TODO you can ask a question with self.question(D=D)
		
		
		
        return 0
        
    def get_branches(self):
        return (self.true_branch, self.false_branch)
    
    def get_depth(self):
        return self.depth