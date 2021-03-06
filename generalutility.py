import json
import numpy as np
from tabulate import tabulate
import os
import matplotlib.pyplot as plt

#Source: Andy Keene - https://github.com/andy-keene/ML/blob/master/perceptron/helpers/helper.py
def print_matrix_as_table(file_name):
    '''
    Prints the confusion matrix of a json obj as a pretty table
    Args:
        file_name (string): file path of .json data, must contain 'confusion_matrix' key
    Returns:
        None
    '''
    headers = [str(x) for x in range(0,10)]
    file_json = dict()

    with open(file_name, 'r') as f:
        file_json = json.load(f)
    pretty_table = tabulate(np.array(file_json['confusion_matrix']), headers, tablefmt="grid")
    print(pretty_table)


#Source: Andy Keene - https://github.com/andy-keene/ML/blob/master/perceptron/helpers/helper.py
def save_data(directory, file_name, results):
    '''
    Saves performance data to:
        directory/file_name.json: raw data
        directory/file_name.png: plot of accuracy over epochs
    '''
    epochs = []
    test_accuracy = []
    train_accuracy = []

    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name + '.json', 'w') as f:
        json.dump(results, f)
    for key in results['accuracy']:
        epochs.append(key)
        train_accuracy.append(results['accuracy'][key]['train'])
        test_accuracy.append(results['accuracy'][key]['test'])

    plt.plot(epochs, train_accuracy)
    plt.plot(epochs, test_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training data', 'test data'], loc='upper left')
    plt.ylim(min(train_accuracy + test_accuracy) - 0.1, max(train_accuracy + test_accuracy) + 0.1)
    plt.savefig(file_name + '.png', bbox_inches='tight')
    plt.clf()