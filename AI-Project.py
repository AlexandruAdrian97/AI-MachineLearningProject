# coding: utf-8

import numpy as np
import re
import os
from collections import defaultdict
from sklearn import svm

PRIMELE_N_CUVINTE = 3000

def accuracy(y, p):
    return 100 * (y == p).astype('int').mean()


def files_in_folder(mypath):
    fisiere = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            fisiere.append(os.path.join(mypath, f))
    return sorted(fisiere)


def extrage_fisier_fara_extensie(cale_catre_fisier):
    nume_fisier = os.path.basename(cale_catre_fisier)
    nume_fisier_fara_extensie = nume_fisier.replace('.txt', '')
    return nume_fisier_fara_extensie


def citeste_texte_din_director(cale):
    date_text = []
    iduri_text = []
    for fis in files_in_folder(cale):
        id_fis = extrage_fisier_fara_extensie(fis)
        iduri_text.append(id_fis)
        with open(fis, 'r', encoding='utf-8') as fin:
            text = fin.read()

        # incercati cu si fara punctuatie sau cu lowercase
        text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        cuvinte_text = text_fara_punct.split()
        date_text.append(cuvinte_text)
    return (iduri_text, date_text)

# citim datele

dir_path = 'C:/Users/Alexandru Adrian/Documents/trainData/'

labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

train_data_path = os.path.join(dir_path, 'trainExamples')

iduri_train, data = citeste_texte_din_director(train_data_path)

print(data[0][:10])

# numaram cuvintele din toate documentele

contor_cuvinte = defaultdict(int)
for doc in data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1', frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)

# extragem primele n cele mai frecvente cuvinte din toate textele
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

print("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])

list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)


### numaram cuvintele din toate documentele ###
def get_bow(text, lista_de_cuvinte):

    contor = dict()
    cuvinte = set(lista_de_cuvinte)
    for cuvant in cuvinte:
        contor[cuvant] = 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1
    return contor


def get_bow_pe_corpus(corpus, lista):
    bow = np.zeros((len(corpus), len(lista)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, lista)
        v = np.array(list(bow_dict.values()))
        # incercati si alte tipuri de normalizari
        v = (v - np.mean(v)) / np.std(v)
        bow[idx] = v
    return bow


def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("Id,Prediction\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(id_text + ',' + str(int(pred)) + '\n')


data_bow = get_bow_pe_corpus(data, list_of_selected_words)
print("Data bow are shape: ", data_bow.shape)

nr_exemple_train = 2979
nr_exemple_valid = 2
nr_exemple_test = len(data) - (nr_exemple_train + nr_exemple_valid)

indici_train = np.arange(0, nr_exemple_train)
indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))

indici_train_valid = np.concatenate([indici_train, indici_valid])

clasificator = svm.SVC(C=0.1, kernel='linear')

clasificator.fit(data_bow[indici_train_valid, :], labels[indici_train_valid])

predictii = clasificator.predict(data_bow[indici_test])

print(f"Acuratete pe test cu C = 0.1:  {accuracy(predictii, labels[indici_test])}%")

importantData_path = os.path.join('C:/Users/Alexandru Adrian/Documents/trainData/testData-public')

testId, importantData = citeste_texte_din_director(importantData_path)

indici = np.arange(0, 1497)

data_test_bow = get_bow_pe_corpus(importantData, list_of_selected_words)

predictii = clasificator.predict(data_test_bow[indici, :])

scrie_fisier_submission("submisie_Kaggle_22.csv", predictii, testId)
