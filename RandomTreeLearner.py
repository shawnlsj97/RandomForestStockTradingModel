""""""
"""  		  	   		 	   		  		  		    	 		 		   		 		  
An implementation of a Random Tree Learner.  	   		 	   		  		  		    	 		 		   		 		  
"""

import numpy as np


class RandomTreeLearner(object):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    This is a Random Tree Learner.

    :param leaf_size: Is the maximum number of samples to be aggregated at a leaf
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		 	   		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """

    def __init__(self, leaf_size=1, verbose=False):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Constructor method  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        self.tree = None
        self.leaf_size = leaf_size
        self.verbose = verbose

    def add_evidence(self, data_x, data_y):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		 	   		  		  		    	 		 		   		 		  

        :param data_x: A set of feature values used to train the learner  		  	   		 	   		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		 	   		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        self.tree = self.build_tree(data_x, data_y)
        if self.verbose:
            print(self.tree)

    def query(self, points):
        """  		  	   		 	   		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		 	   		  		  		    	 		 		   		 		  

        :param points: A numpy array with each row corresponding to a specific query.  		  	   		 	   		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		 	   		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		 	   		  		  		    	 		 		   		 		  
        """
        predicted = np.zeros(points.shape[0])
        for idx, row in enumerate(points):
            # start from root
            curr_idx = 0
            curr_node = self.tree[curr_idx]
            while curr_node[0] != -1:
                factor_val = row[int(curr_node[0])]  # curr_node[0] indicates factor
                split_val = curr_node[1]
                if factor_val <= split_val:
                    curr_idx += int(curr_node[2])
                else:
                    curr_idx += int(curr_node[3])
                curr_node = self.tree[curr_idx]

            predicted[idx] = curr_node[1]
        return predicted

    def build_tree(self, dataX, dataY):
        """
        Constructs a Random Tree from the given data X and data Y.
        A Random Tree is represented internally as a NumPy array with each row containing 4 columns:
            1) Factor
            2) Split Value
            3) Left (relative)
            4) Right (relative)

        :param dataX: A set of feature values used to train the learner
        :type dataX: numpy.ndarray
        :param dataY: The value we are attempting to predict given the X data
        :type dataY: numpy.ndarray
        :return: A Random Tree from the given data X and data Y
        :rtype: RandomTree
        """
        if dataX.shape[0] == 1:
            return np.array([[-1, dataY[0], -1, -1]])
        if dataX.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(dataY), -1, -1]])
        if np.all(dataY == dataY[0]):
            return np.array([[-1, dataY[0], -1, -1]])
        else:
            split_feature_col_idx = self.determine_split_feature(dataX, dataY)
            split_feature_col = dataX[:, split_feature_col_idx]
            split_val = np.median(split_feature_col)
            if (np.array_equal(dataX[split_feature_col <= split_val], dataX) or
                    np.array_equal(dataX[split_feature_col > split_val], dataX)):
                return np.array([[-1, np.mean(dataY), -1, -1]])
            left_tree = self.build_tree(dataX[split_feature_col <= split_val], dataY[split_feature_col <= split_val])
            right_tree = self.build_tree(dataX[split_feature_col > split_val], dataY[split_feature_col > split_val])
            root = np.array([[split_feature_col_idx, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))

    def determine_split_feature(self, data_x, data_y):
        """
        Choose feature (Xi) randomly

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        :return: Index of the chosen feature
        :rtype: int
        """
        return np.random.choice(data_x.shape[1])
