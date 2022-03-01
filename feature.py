import sys
import csv
import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

# setting up global constant
assert(len(sys.argv) == 10)
train_input = sys.argv[1]
validation_input = sys.argv[2]
test_input = sys.argv[3]
dict_input = sys.argv[4]
feature_dictionary_input = sys.argv[5]
formatted_train_out = sys.argv[6]
formatted_validation_out = sys.argv[7]
formatted_test_out = sys.argv[8]
feature_flag = sys.argv[9]


# model 1: convert dataset of review to an array of vectors
def dataset2vec1(dataset, dictionary):
    res = np.array() 
    for review in dataset:
        feature_vec = np.zeros(len(dictionary)+1)
        feature_vec[0] = review[0]
        for word, index in dictionary:
            if word in review[1]:
                feature_vec[index+1] = 1
        res = np.vstack((res, feature_vec))
    return res

# model 2: convert dataset of review to an array of vectors
def dataset2vec2(dataset, dictionary):
    res = np.empty([0, VECTOR_LEN+1]) # how to create an empty array?
    for review in dataset:
        temp = np.array([review[0]]) # how to convert label to a 6 sig digit float?
        sum_vec = np.zeros(VECTOR_LEN)
        count = 0
        for word in review[1]:
            if word in dictionary.keys():
                np.add(sum_vec, dictionary[word])
                count += 1
        feature_vec = np.hstack((temp, sum_vec/count)) # how to convert label to a 6 sig digit float?
        res = np.vstack((res, feature_vec))
    return res

#np.savetxt
def array2tsv(arr, path):
    with open(path, 'w') as f_out:
        for vector in arr:
            for number in vector:
                f_out.write(str(number) + '\t')
            f_out.write(str(number) + '\n')
    return

# organize review dataset to feature vectors
def feature(train_input, validation_input, test_input, dict_input, feature_dictionary_input, formatted_train_out, formatted_validation_out, formatted_test_out, feature_flag):
    train_dataset = load_tsv_dataset(train_input)
    validation_dataset = load_tsv_dataset(validation_input)
    test_dataset = load_tsv_dataset(test_input)
    # bag of words
    if feature_flag == 1:
        bag_dict = load_dictionary(dict_input)
        train_feature_array = dataset2vec1(train_dataset, bag_dict)
        validation_feature_array = dataset2vec1(validation_dataset, bag_dict)
        test_feature_array = dataset2vec1(test_dataset, bag_dict)
        np.savetxt(formatted_train_out, train_feature_array, delimiter='\t', fmt='%d')
        np.savetxt(formatted_validation_out, validation_feature_array, delimiter='\t', fmt='%d')
        np.savetxt(formatted_test_out, test_feature_array, delimiter='\t', fmt='%d')

    # feature dictionary
    else:
        feature_dict = load_feature_dictionary(feature_dictionary_input)
        train_feature_array = dataset2vec2(train_dataset, feature_dict)
        validation_feature_array = dataset2vec2(validation_dataset, feature_dict)
        test_feature_array = dataset2vec2(test_dataset, feature_dict) 
        np.savetxt(formatted_train_out, train_feature_array, delimiter='\t', fmt='%.6f')
        np.savetxt(formatted_validation_out, validation_feature_array, delimiter='\t', fmt='%.6f')
        np.savetxt(formatted_test_out, test_feature_array, delimiter='\t', fmt='%.6f')           
       


    return

if __name__ == '__main__':
    feature(train_input, validation_input, test_input, dict_input, feature_dictionary_input, formatted_train_out, formatted_validation_out, formatted_test_out, feature_flag)