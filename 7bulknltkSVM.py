#import regex
import re
import csv
import pprint
import nltk.classify
# import svm
# from svmutil import *
import pickle, os
# import classifier_helper, html_helper
from sklearn import svm

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features
#end


#Read the tweets one by one and process it
inpTweets = csv.reader(open('data/sampleTweetsID.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('data/feature_list/stopwordsID.txt')
count = 0;
featureList = []
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));
#end loop

def getSVMFeatureVectorAndLabels(tweets, featureList):
    # print tweets
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        # print t
        label = 0
        map = {}
        #Initialize empty map
        for w in sortedFeatures:
            map[w] = 0
        
        tweet_words = t[0]
        # print tweet_words
        tweet_opinion = t[1]
        # print tweet_opinion
        # print ''
        #Fill the map
        for word in tweet_words:
            word = replaceTwoOrMore(word) 
            word = word.strip('\'"?,.')
            # print word
            if word in map:
                map[word] = 1
        # print map
        #end for loop
        values = map.values()
        feature_vector.append(values)
        # print feature_vector
        if(tweet_opinion == 'positive'):
            label = 0
        elif(tweet_opinion == 'negative'):
            label = 1
        elif(tweet_opinion == 'neutral'):
            label = 2
        labels.append(label)
        # print labels
    return {'feature_vector' : feature_vector, 'labels': labels}
#end

#start getSVMFeatureVector
def getSVMFeatureVector(tweets, featureList):
    # print tweets
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []

    #Initialize empty map
    for w in sortedFeatures:
        map[w] = 0
    # print map
        
    for t in tweets:
        # print t
        # label = 0
        # map = {}
        
        #Fill the map
        # print map
        # print t
        word = t
        # print t
        # print word
        # for word in t:
            # print word
        if word in map:
            # print 'ada'
            map[word] = 1
        # print map
        #end for loop
        values = map.values()
        # feature_vector.append(values)
        # print feature_vector
    # feature_vector.append(values)
    feature_vector = values
    # print feature_vector
    # print values
    return feature_vector
#end

# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, tweets)

testTweet = 'Hari yang mengecewakan. Menghadiri pameran mobil untuk mencari pendanaan, harganya malah lebih mahal'
test_tweets2 = processTweet(testTweet)
test_tweets = getFeatureVector(test_tweets2, stopWords)
test_feature_vector = getSVMFeatureVector(test_tweets, featureList)
#Train the classifier
# print test_feature_vector
result = getSVMFeatureVectorAndLabels(tweets, featureList)
# print result

### dari sini
# problem = svm_problem(result['labels'], result['feature_vector'])
# # print problem
# #'-q' option suppress console output
# param = svm_parameter('-q')
# param.kernel_type = LINEAR
# classifier = svm_train(problem, param)
# # svm_save_model(classifierDumpFile, classifier)
# #Test the classifier
# test_feature_vector = getSVMFeatureVector(test_tweets, featureList)
# #p_labels contains the final labeling result
# p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)
### sampai sini

klasifikasi = svm.SVC()
# klasifikasi = svm.LinearSVC()
# klasifikasi = svm.SVR()
# print result['labels']
klasifikasi = klasifikasi.fit(result['feature_vector'],result['labels'])

prediksi = klasifikasi.predict([test_feature_vector])
