import re
import random
from random import randint
from numpy import asarray
from keras.preprocessing.sequence import pad_sequences

# Warning: Normally its '  but in the text is ’
def decontracted(phrase):
    phrase = re.sub(r"wo n\’t", "will not", phrase)
    phrase = re.sub(r"ca n\’t", "can not", phrase)
    phrase = re.sub(r"wont","will not", phrase)
    phrase = re.sub(r"cant", "can not", phrase)
    phrase = re.sub(r"wouldnt", "would not", phrase)
    phrase = re.sub(r"couldnt","could not",phrase)
    # general
    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)
    return phrase

def generateMaskedSentence(processedSentence,frequentDic):
    print("Generate Maksed Sentence")
    for sentence in processedSentence:
        for i in range(len(sentence)):
            if sentence[i] not in frequentDic.keys():
                sentence[i] = marked_token
    return processedSentence

def generateIndexForDataset(dataSize):
    print("Generate Index For Dataset")
    trainIndexList = []
    data_index = []

    for i in range(dataSize-1):
        x = randint(0,dataSize-1)
        y = randint(0,dataSize-1)
        z = i+1 
        while(x == y or x==i or x==z or y == i or y==z ):
            x = randint(0,dataSize-1)
            y = randint(0,dataSize-1)
        index = ([[x,0],[y,0],[z,1]])
        random.shuffle(index)
        trainIndexList.append([i,index])   

    for row in trainIndexList:
        train_x = row[0]
        y_index = row[1]
        y_output = []
        x_masked = []
        for index in y_index:
            y_output.append(index[1])
            x_masked.append(index[0])
        data_index.append([train_x,x_masked,y_output])

    print("Finish Generating index")
    return data_index

def loadFile(textPath,orgFile):
    print("Load:" + textPath + orgFile)
    with open(textPath +  orgFile ,encoding="utf-8") as fp:
        lines = fp.readlines()
    lines = [e for e in lines if e not in {'\n'}]
    return lines
    
def preprocessing(lines): 
    print("Pre processing Data")
    count = 0 
    for line in lines:
        # Decapitalize: Conver to lower case first. 
        line = line.lower()
        # Delete url 
        line = re.sub(r"http\S+", "link", line)
        line = re.sub(r"\S+html", "link", line)
        line = re.sub(r"\S+.com$", "link", line)
        line = re.sub(r"\S+.jpg$", "photo", line)
        line = decontracted(line)
        '''
        ignore:  * { } \  < > 
        Splite based on : ! . ,  &  # ' $ 
        line = re.findall(r"[\w']+|[().,:!?;'$&]", line)
        '''
        lines[count] = line
        count += 1 
    return lines

def getVocabFrequencyForText(text, dic):
    for sent in text:
        for vocab in sent:
            if vocab not in dic.keys():
                dic[vocab] = 1
            else: dic[vocab] += 1
    return dic

def sentenceToWordList(lines):
    print("Convert sentence to word list")
    count = 0
    for line in lines:
        lines[count] = re.findall(r"[\w']+|[().,:!?;'$&]", line)
        count += 1
    return lines


def sentenceToTokenData(textPath,orgFile,t,dic):
    processedLine = loadFile(textPath,orgFile)
    processedLine = sentenceToWordList(processedLine)
    # print("Deep copy")
    maskedLine = copy.deepcopy(processedLine)
    maskedLine =  generateMaskedSentence(maskedLine,dic)
    testIndex = randint(0,100)
    print("--Test-- Process:{} ; masked:{}".format(processedLine[testIndex],maskedLine[testIndex]))
  
    processedLine = t.texts_to_sequences(processedLine)
    maskedLine = t.texts_to_sequences(maskedLine)

    processedLine = pad_sequences(processedLine, maxlen=maxLength, padding='post')
    maskedLine = pad_sequences(maskedLine, maxlen=maxLength, padding='post')
    return processedLine,maskedLine 

def generateTrainingDataSet(processedLine,maskedLine):
    dataIndex = generateIndexForDataset(len(processedLine))
    # Modeling
    input_X = []
    input_masked_first = []
    input_masked_second = []
    input_masked_third = []
    input_Y = []
    for index in dataIndex:
        input_X.append(processedLine[index[0]])
        input_Y.append(index[2])
        input_masked_first.append(maskedLine[index[1][0]])
        input_masked_second.append(maskedLine[index[1][1]])
        input_masked_third.append(maskedLine[index[1][2]])
    input_X = asarray(input_X)
    input_masked_first = asarray(input_masked_first)
    input_masked_second = asarray(input_masked_second)
    input_masked_third = asarray(input_masked_third)
    input_Y = asarray(input_Y)
    return input_X,input_masked_first,input_masked_second,input_masked_third,input_Y

