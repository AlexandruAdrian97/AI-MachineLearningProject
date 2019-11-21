# coding: utf-8

import numpy as np
import re
import os
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import time

start = time.time()

first_n_words = 20000
director_path = 'C:/Users/Alexandru Adrian/Documents/trainData/'
print('')


def accuracy(y, p):
    return 100 * (y == p).astype('int').mean()


def files_in_folder(my_path):
    files = []
    for f in os.listdir(my_path):
        if os.path.isfile(os.path.join(my_path, f)):
            files.append(os.path.join(my_path, f))
    return sorted(files)


def extract_file_without_extension(path_to_file):
    file_name = os.path.basename(path_to_file)
    file_name_without_extension = file_name.replace('.txt', '')
    return file_name_without_extension


def read_texts_from_director(path):
    text_data = []
    text_id = []
    for file in files_in_folder(path):
        id_file = extract_file_without_extension(file)
        text_id.append(id_file)
        with open(file, 'r', encoding='utf-8') as fin:
            text = fin.read()

        text_without_punctuation = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        words_from_text = text_without_punctuation.split()
        text_data.append(words_from_text)
    return (text_id, text_data)


def get_bow(text, list_of_words):
    counter = dict()
    words = set(list_of_words)
    for word in words:
        counter[word] = 0
    for word in text:
        if word in words:
            counter[word] += 1
    return counter


def get_bow_on_corpus(corpus, list_a):
    bow = np.zeros((len(corpus), len(list_a)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, list_a)
        v = np.array(list(bow_dict.values()))
        v = (v - np.mean(v)) / np.std(v)
        bow[idx] = v
    return bow


def get_submission_file(file_name, prediction, ids):
    with open(file_name, 'w') as fout:
        fout.write("Id,Prediction\n")
        for text_id, pred in zip(ids, prediction):
            fout.write(text_id + ',' + str(int(pred)) + '\n')


labels = np.loadtxt(os.path.join(director_path, 'labels_train.txt'))

train_data_path = os.path.join(director_path, 'trainExamples')
train_id, train_data = read_texts_from_director(train_data_path)

test_data_path = os.path.join(director_path, 'testData-public')
test_Id, test_data = read_texts_from_director(test_data_path)

words_counter = defaultdict(int)
for doc in train_data:
    for word in doc:
        words_counter[word] += 1

words_with_frequency = list(words_counter.items())
words_with_frequency = sorted(words_with_frequency, key=lambda kv: kv[1], reverse=True)
words_with_frequency = words_with_frequency[0:first_n_words]

list_of_selected_words = []
for word, frequency in words_with_frequency:
    list_of_selected_words.append(word)

data_bow = get_bow_on_corpus(train_data, list_of_selected_words)
test_data_bow = get_bow_on_corpus(test_data, list_of_selected_words)

number_examples_train = 2700
number_examples_validation = 150
number_examples_test = len(train_data) - (number_examples_train + number_examples_validation)

train_indices = np.arange(0, number_examples_train)
validation_indices = np.arange(number_examples_train, number_examples_train + number_examples_validation)
test_indices = np.arange(number_examples_train + number_examples_validation, len(train_data))
train_validation_indices = np.concatenate([train_indices, validation_indices])


# VALIDATION

classifier_MLP = MLPClassifier(hidden_layer_sizes=(300, 200), activation='tanh', solver='adam', alpha=0.001, max_iter=200)

classifier_MLP.fit(data_bow[train_indices, :], labels[train_indices])

predictions = classifier_MLP.predict(data_bow[validation_indices, :])

print(f"Accuracy with MPL classifier on validation: {accuracy(predictions, labels[validation_indices])} %")
print('')


# TEST
classifier_MLP.fit(data_bow[train_validation_indices, :], labels[train_validation_indices])

predictions = classifier_MLP.predict(data_bow[test_indices])

print(f"Accuracy with MPL pe test: {accuracy(predictions, labels[test_indices])} %")
print(' ')

test_confusion_matrix = confusion_matrix(labels[test_indices], predictions)
print(test_confusion_matrix)

scores = []
cv = KFold(n_splits=10)
for train_validation_indices, test_indices in cv.split(test_data_bow):
    print("Train Indices: ", train_validation_indices, "\n")
    print("Test Indices: ", test_indices, "\n")
    print(' ')

    x_train, x_test, y_train, y_test = data_bow[train_validation_indices], data_bow[test_indices], labels[train_validation_indices], labels[
        test_indices]
    classifier_MLP.fit(x_train, y_train)
    scores.append(classifier_MLP.score(x_test, y_test))
print(f'The kfold cv score is: {np.mean(scores)}')




# VALIDATION SVM:

for C in [0.01, 0.1, 1, 10, 100]:
    classifier_svm = svm.SVC(C=C, kernel='linear')
    classifier_svm.fit(data_bow[train_indices, :], labels[train_indices])
    svm_predictions = classifier_svm.predict(data_bow[validation_indices, :])
    print(f"Accuracy on validation with C = {C}: { accuracy(svm_predictions, labels[validation_indices])} %")


# TEST SVM:
print(' ')
for CC in [0.01, 0.1, 1, 10, 100]:
    classifier_svm = svm.SVC(C=CC, kernel='linear')
    classifier_svm.fit(data_bow[train_validation_indices, :], labels[train_validation_indices])
    predictions = classifier_svm.predict(data_bow[test_indices])
    print(f"Accuracy on test with C = {CC}: {accuracy(predictions, labels[test_indices])} %")


classifier_MLP.fit(data_bow[train_validation_indices, :], labels[train_validation_indices])
Prediction = classifier_MLP.predict(test_data_bow)
end = time.time()

get_submission_file("submisie_Kaggle_MLP_312.csv", Prediction, test_Id)


print('Time: ', int(end - start), 'seconds')
