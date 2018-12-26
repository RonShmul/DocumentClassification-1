import os
import pandas as pd
import numpy
from os import listdir
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt

path = "D:\\documents\\users\\avevanes\\Downloads\\ohsumed-first-20000-docs"

# declaration
tokenize = []

# files name inside a folder
trainFileDirs = []
testFileDirs = []

# array of all the data
trainData = []
testData = []

infoTable = []

categoties = []
docForCategory = []
termsForCategory = []


from nltk.corpus import stopwords

stop_words = stopwords.words('english')

def cleanAndNormalizeText(data):

    # tokenization
    tokenize = word_tokenize(data)
    # remove stop words
    filterText = [w for w in tokenize if not w in stop_words]
    filterText = [w for w in filterText if not len(w) <= 1]
    # stem
    ps = PorterStemmer()
    for i in range(len(filterText)-1):
        if len(filterText[i]) > 1:
            try:
                filterText[i] = ps.stem(filterText[i])
            except Exception as e:
                filterText[i] = filterText[i]
    return ' '.join(filterText)

trainFolder = os.listdir(path+ '\\training')
testFolder = os.listdir(path+ '\\test')
for folder in trainFolder:
    trainFileDirs = os.listdir(path+ '\\training\\'+ folder)
    for file in trainFileDirs:
        fileReader = open(path+ '\\training'+'\\'+folder+'\\'+file,mode='r')
        text = fileReader.read().lower()
        fileReader.close()
        Text = cleanAndNormalizeText(text)
        infoTable.append(['train',folder,Text, file])
for folder in testFolder:
    testFileDirs = os.listdir(path+ '\\test\\'+ folder)
    for file in testFileDirs:
        fileReader = open(path+ '\\test'+'\\'+folder+'\\'+file,mode='r')
        text = fileReader.read().lower()
        fileReader.close()
        Text = cleanAndNormalizeText(text)
        # infoTable.append(['test',folder, file, Text])


trainTable = pd.DataFrame(infoTable, columns=['type','category','text','file'])

# term distribution per category
print('Number of categories', numpy.unique (trainTable['category']).size)

# category and num of docs in it
docForCategory = trainTable.groupby(by='category', as_index=False).agg({'file': pd.Series.nunique})
docForCategory

# term distribution per category
categories = trainTable['category']
for cat in categories:
    categoryFiles = trainTable.loc[trainTable['category'] == cat]
    categoryData = [" ".join(set(txt.split(' '))) for txt in categoryFiles['text']]
    categoryData = " ".join(categoryData)
    termsCount = pd.Series(categoryData.split()).value_counts()
    categoryTermValues = pd.DataFrame
    categoryTermValues['term'] = termsCount.index
    categoryTermValues['count'] = list(termsCount)
    setCount=set(categoryTermValues['count'])
    his = categoryTermValues.hist(columns='count',bins=len(setCount))
    plt.xlabel("number of files")
    plt.ylabel("number of terms")
    plt.show(his)


