# Artificial-Intelligence-ProjectML

The program is written in Python and it is a project developed for the Artificial Intelligence course at University.

I have a train data folder with over 2800 texts, translated from 11 different languages into English as examples.

The program takes from the train data 80% of the texts and turn them into training data and the other 20% is used for testing.

After the training is done, the program will know the labels for each of the train data texts and the algorithm will take the most 20000 frequent words from all the train data texts and try to classify the test data based on what it learned.

I used two classifier for this program, Neural Networks(MLP) and SVM. The MLP scored a better accuracy therefore I chose this classifier for the final result.

I used a K-FOLD Cross-Validation to improve my accuracy.

Finally, the algorithm was able to get over 90% accuracy.




