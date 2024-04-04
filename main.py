from collections import Counter
import random
import re
import os


class Perceptron:
    """ A class to represent a Perceptron
    
        ...
       
        Attributes
        ----------
        alpha : float
            learning constant
        weights : list
            vector (as list) of weights
        activating_result : str
            name of the class that should activate perceptron
        
        Methods
        -------
        __set_weights(number_of_attributes):
            Sets random weights
        compute(vector) : 
            Computes net value and returns verdict for given vector
        learn(good_result, vector) : 
            Modifies weights and threshold using delta funcition

    """

    def __init__(self, alpha, activating_result, number_of_attributes) :
        """Contstructs all necessary attributes
        
            Args: 
                alpha(float): learning constant
                activating_result(str) : name of the class that should activate perceptron
                number_of_attributes(int): number of entries of perceptron
        """
        
        self.alpha = alpha
        self.weights = []
        self.activating_result = activating_result
        self.__set_weights(number_of_attributes)
        
    def __set_weights(self, number_of_attributes):
        """Sets begining weights as random generated numbers

        Args:
            number_of_attributes (int): number of weights to generate (without threshold)
        """        
        
        for i in range(number_of_attributes + 1):
            self.weights.append(random.randint(-5, 5))

    def compute(self, vector):
        """Computes output of perceptron

        Args:
            vector (list): vector to compute output from

        Returns:
            int: 1 activated or 0 if not
        """        
        
        vector.append(-1)
        net = 0
        
        for i in range(len(vector)):
            net += (float(vector[i]) * self.weights[i])
        if net >= 0:
            return 1
        
        return 0

    def learn(self, good_result, vector):
        """Function that teaches percpetron by changing weights and threshold using delta function

        Args:
            good_result (int): 0 or 1 depanding on the result given in train_set in Trainer
            vector (list): vector to compute and learn on
        """        
        prev_result = self.compute(vector)
        
        vector.append(-1)
        
        passed_result = 0
        if good_result == self.activating_result: passed_result = 1

        for i in range(len(self.weights)):
            
            self.weights[i] += ((passed_result - prev_result)
                                * self.alpha 
                                * float(vector[i]))
            

class NeuralNetwork:
    def __init__(self, alpha, classes, number_of_attributes):
        self.perceptrons = self.__create_neural_network()
        self.alpha = alpha
        self.classes = classes
        self.number_of_attributes = number_of_attributes
    
    def __create_neural_network(self):
        network = []
        
        for i in range(len(self.classes)):
            network.append(Perceptron(self.alpha, self.classes[i], self.number_of_attributes))
class Trainer:
    """A class to train the perceptron
    
        ...
       
        Attributes
        ----------
        percpeptron : Perceptron
            perceptron to train
        train_set_fname : str
            name of train_set file
        train_set : list
            list of vector to train on
        names_of_classes : list
            list of names of classes in dataset
        
        Methods
        -------
        __set_names_of_classes() :
            Finds and sets names of classes from dataset in correct order
        train(number_of_trainings) :
            Method to train perceptron on given train_set
        
    """    
    
    def __init__(self, neural_network, train_set): 
        """Contructor of Trainer class
        

        Args:
            perceptron (Perceptron): perceptron to train passed from ui
            train_set_fname (str): name of file containing train_set
        """        
        self.neural_network = neural_network
        self.names_of_classes = neural_network.classes   
        self.train_set = train_set
        
        random.shuffle(self.train_set)
        

    def train(self, number_of_trainings):
        """Method that trains perceptron

        Args:
            number_of_trainings (int): number of passes perceptron has to do through train_set
        """        
        for perceptron in self.neural_network:
            for i in range(number_of_trainings):
                for line in self.train_set:
                    perceptron.learn(line[-1], line[:-1])

class _DataSetCreator:
    
    @staticmethod
    def __normalize(dataset):
        attributes_values = []

        #get values for each attribute in dataset
        for i in range(len(dataset[0])-1):
            attributes_values.append([])
        for row in dataset:
            for i in range(len(row)-1):
                attributes_values[i].append(float(row[i]))

        max_attributes = []
        min_attributes = []

        #get max and min value for each attribute in dataset
        for attribute in attributes_values:
            max_attributes.append(max(attribute))
            min_attributes.append(min(attribute))

        #normalize every value
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = round((float(row[i]) - min_attributes[i])/(max_attributes[i] - min_attributes[i]),3)
        
        return dataset
    
    @staticmethod
    def create_vector_list(dir_name):
        vector_list = []
        rootdir = os.getcwd() + "/" + dir_name
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                vector = [0]*26
                language = os.path.basename(subdir)
                file = open(os.path.join(subdir,file),"r")
                text = file.read().strip()
                regex = re.compile('[^a-zA-Z]+')
                text = regex.sub('',text).lower()
                letter_counter = Counter(text)
                number_of_letters = letter_counter.total()
                for i in range(26):
                    vector[i] = letter_counter[chr(i+ord('a'))]/number_of_letters
                vector.append(language)
                vector_list.append(vector)
                
        vector_list = _DataSetCreator.__normalize(vector_list)
        
        return vector_list
    
    @staticmethod
    def get_names_of_classes(data):
        """Sets list of names of classes (index 0 nonactivating, 1 - activating)

        Returns:
            list: list of names of classes
        """        
        
        classes = []
        
        for line in data:
            if line[-1] not in classes:
                classes.append(line[-1])
        
        return classes
    
    
        
data_set = _DataSetCreator.create_vector_list("data")
classes = _DataSetCreator.get_names_of_classes(data_set)
Trainer(NeuralNetwork(0.5, classes, len(data_set[0])-1), data_set)
    
