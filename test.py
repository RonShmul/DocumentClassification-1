import os
import pandas as pd
from os import listdir
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

path = "C:\\Users\\avevanes\\ohsumed-first-20000-docs"

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

    # stem
    ps = PorterStemmer()
    for i in range(len(filterText)-1):
        if len(filterText[i]) > 1:
             filterText[i] = ps.stem(filterText[i])
    return filterText


trainFolder = os.listdir(path+ '\\training')
testFolder = os.listdir(path+ '\\test')
for folder in trainFolder:
    trainFileDirs = os.listdir(path+ '\\training\\'+ folder)
    for file in trainFileDirs:
        fileReader = open(path+ '\\training'+'\\'+folder+'\\'+file,mode='r')
        text = fileReader.read().lower()
        fileReader.close()
        Text = cleanAndNormalizeText(text)
        infoTable.append(['train',folder, file, Text])
for folder in testFolder:
    testFileDirs = os.listdir(path+ '\\test\\'+ folder)
    for file in testFileDirs:
        fileReader = open(path+ '\\test'+'\\'+folder+'\\'+file,mode='r')
        text = fileReader.read().lower()
        fileReader.close()
        Text = cleanAndNormalizeText(text)
        infoTable.append(['test',folder, file, Text])


table = pd.DataFrame(infoTable, columns=['type','category','test','file'])
table