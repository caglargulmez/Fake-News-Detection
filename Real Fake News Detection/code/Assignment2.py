
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import nltk
from nltk.stem.snowball import SnowballStemmer


lemma = nltk.wordnet.WordNetLemmatizer()
stemmer = SnowballStemmer('english')

# read file, return an array which holds all line, each line is one string
def read(path):
    file = open(path, 'r')
    return [line.rstrip('\n') for line in file.readlines()]


# reading with nltk's lemmatizer
def read2(path):
    file = open(path, 'r')
    news = []
    for line in file.readlines():
        sentence = ""
        line = [lemma.lemmatize(word) for word in line.rstrip("\n").split(" ")]

        for i in range(len(line)):
            if not line[i].isdigit():
                sentence += line[i] + " "

        news.append(sentence.rstrip(' '))
    return news


#reading with nltk's snowball stemmer
def read3(path):
    file = open(path, 'r')
    news = []
    for line in file.readlines():
        sentence = ""
        line = [stemmer.stem(word) for word in line.rstrip("\n").split(" ")]

        for i in range(len(line)):
            if not line[i].isdigit():
                sentence += line[i] + " "

        news.append(sentence.rstrip(' '))
    return news

# uni, bi


def create_vectorizer(arr, ngram):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(ngram, ngram))
    X = vectorizer.fit_transform(arr).toarray()
    X = np.array(np.sum(X, axis=0))
    d1 = vectorizer.fit(arr).vocabulary_
    return X, d1


def fake_probability(word, arr1, arr2, d1, d2):

    fake_index = d1.get(word, 0.0)
    real_index = d2.get(word, 0.0)

    if fake_index == 0 and real_index == 0:
        return 1
    elif fake_index == 0 and real_index != 0:
        return 1 / (1 + arr2[real_index])
    elif fake_index != 0 and real_index == 0:
        return arr1[fake_index] / (1 + arr1[fake_index])
    else:
        return arr1[fake_index] / (arr1[fake_index] + arr2[real_index])


def real_probability(word, arr1, arr2, d1, d2):

    fake_index = d1.get(word, 0.0)
    real_index = d2.get(word, 0.0)

    if fake_index == 0 and real_index == 0:
        return 1
    elif fake_index == 0 and real_index != 0:
        return arr2[real_index] / (1 + arr2[real_index])
    elif fake_index != 0 and real_index == 0:
        return 1 / (1 + arr1[fake_index])
    else:
        return arr2[real_index] / (arr1[fake_index] + arr2[real_index])



def naive_bayes(sentences_tuple, arr1, arr2, d1, d2, fake_freq, real_freq, ngram):
    print('Naive Bayes process started.\n------------------------------------')
    # sentences_tuple  : test data to predict
    #            arr1  : real data train
    #            arr2  : fake data train
    #              d1  : fake data dictionary
    #              d2  : real data dictionary
    #       fake_freq  : (fake line count) / (fake + real)
    #       real_freq  : (real line count) / (fake + real)
    #           ngram  : if 1 -> unigram, if 2 -> bigram
    true, false = 0, 0
    real_prob, fake_prob = 0, 0

    file_unigram = open('test_unigram2.csv', 'w+')
    file_bigram = open('test_bigram.csv', 'w+')

    if ngram == 1:
        for pair in sentences_tuple:
            real_prob = 0
            fake_prob = 0
            prediction = ''
            for word in pair[0].split(' '):
                fake_prob += math.log(fake_probability(word, arr1, arr2, d1, d2))
                real_prob += math.log(real_probability(word, arr1, arr2, d1, d2))
            fake_prob += math.log(fake_freq)
            real_prob += math.log(real_freq)
            #print(real_prob, fake_prob)

            if real_prob > fake_prob:
                #print("Prediction: Real\tActual: " + pair[1])
                prediction = "real"
                file_unigram.write(pair[0] + ',real\n')

            else:
                prediction = "fake"
                #print("Prediction: Fake\tActual: " + pair[1])
                file_unigram.write(pair[0] + ',fake\n')
            if prediction == pair[1]:
                true += 1
            else:
                false += 1
        accuracy = 100 * (true / (true + false))
        print("Unigram->",accuracy)

    else:
        for pair in sentences_tuple:
            real_prob = 0
            fake_prob = 0
            prediction = ''
            line = pair[0].split(' ')
            for i in range(len(line)-1):
                word_couple = line[i] + ' ' + line[i+1]
                #print(word_couple)
                fake_prob += math.log(fake_probability(word_couple, arr1, arr2, d1, d2))
                real_prob += math.log(real_probability(word_couple, arr1, arr2, d1, d2))
            fake_prob += math.log(fake_freq)
            real_prob += math.log(real_freq)
            #print(real_prob, fake_prob)
            if real_prob > fake_prob:
                #print("Prediction: Real\tActual: " + pair[1])
                prediction = "real"
                file_bigram.write(pair[0] + ',real\n')

            else:
                prediction = "fake"
                #print("Prediction: Fake\tActual: " + pair[1])
                file_bigram.write(pair[0] + ',fake\n')

            if prediction == pair[1]:
                true += 1
            else:
                false += 1
        file_bigram.close()
        file_unigram.close()
        aaccuracy = 100 * (true / (true + false))
        print("Bigram->", aaccuracy)

    print('Naive Bayes process finished\n------------------------------------')



def test(path, arr1, arr2, d1, d2, fake_freq, real_freq, ngram):
    file = open(path, 'r')
    file.readline()  # skip one line -> Id, Category

    # sentence_tuple: ( line, true/false)
    sentences_tuple = [(line.rstrip('\n').split(',')[0], line.rstrip('\n').split(',')[1]) for line in file.readlines()]
    if ngram == 1:
        naive_bayes(sentences_tuple, arr1, arr2, d1, d2, fake_freq, real_freq, 1)
    else:
        naive_bayes(sentences_tuple, arr1, arr2, d1, d2, fake_freq, real_freq, 2)


if __name__ == '__main__':

    train_real_path = "./Data/clean_real-Train.txt"
    train_fake_path = "./Data/clean_fake-Train.txt"
    test_path = "./Data/test.csv"

    real_news = read(train_real_path)
    fake_news = read(train_fake_path)
    print("Fake News:", fake_news)
    print("Real News:", real_news)

    # unigram data
    fake_arr1, fake_d1 = create_vectorizer(fake_news, 1)
    real_arr1, real_d1 = create_vectorizer(real_news, 1)

    # bigram data
    fake_arr2, fake_d2 = create_vectorizer(fake_news, 2)
    real_arr2, real_d2 = create_vectorizer(real_news, 2)

    fake_frequency = len(fake_news) / (len(fake_news) + len(real_news))
    real_frequency = len(real_news) / (len(fake_news) + len(real_news))

    test(test_path, fake_arr1, real_arr1, fake_d1, real_d1, fake_frequency, real_frequency, 1)

    test(test_path, fake_arr2, real_arr2, fake_d2, real_d2, fake_frequency, real_frequency, 2)

    print(len(fake_d1))
    print(len(fake_d2))
    print(len(real_d1))
    print(len(real_d2))