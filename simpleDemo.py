#import regex
import re
import csv
import pprint
import nltk.classify
import libsvm_classifier
import baseline_classifier,naive_bayes_classifier,max_entropy_classifier

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

#sampleTweets
#Read the tweets one by one and process it
inpTweets = csv.reader(open('data/sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('data/feature_list/stopwords.txt')
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

# Remove featureList duplicates
featureList = list(set(featureList))

# Generate the training set
training_set = nltk.classify.util.apply_features(extract_features, tweets)

# Train the Naive Bayes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
# Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis
# Test the classifier
testTweet = 'I like you'
processedTestTweet = processTweet(testTweet)
sentiment = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
# print getFeatureVector(processedTestTweet, stopWords)
print "testTweet = %s, sentiment = %s\n" % (testTweet, sentiment)




plainstring2 = dict()
inputTweets = list()
# testTweet = "I like you"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "Can't believe I have to wait another 6 months for my phone contract to end! I'm bored now!!! The 12 month contract would have run out!!! "
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "When did I felt so lonely? "
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "ugh. a huge headache, coughing constantly, legs feeling week, and feeling like throwing up.  This sucks beyond compare "
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "Got to go clean now, knowing it will be messed up again by tomorrow. "
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "is still hoping Google take over the world. Algebra revision "
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "has his new M4400 w/ Core2 Extreme X9100 &amp; SSE4.1 64-bit buildchain finally rebuilt, but also found an igraph/ARPACK regression "
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "@slightsarcasm tom  (time of month ;) ) not your bf tom :L"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "what really surprises about wisegirls is its low-key quality and genuine tenderness ."
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "( wendigo is ) why we go to the cinema : to be fed through the eye , the heart , the mind ."
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "one of the greatest family-oriented , fantasy-adventure movies ever ."
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "ultimately , it ponders the reasons we need stories so much ."
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "an utterly compelling 'who wrote it' in which the reputation of the most famous author who ever lived comes into question ."
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "illuminating if overly talky documentary ."
# inputTweets.append(unicode(testTweet, "utf-8"))
movieComments = open('/Users/PAllAvi/Downloads/documents-export-2016-04-02/Maleficent_update', 'rb')

count = 0
for comment in movieComments:
    # print comment
    inputTweets.append(unicode(comment, "utf-8"))
    count = count + 1
    if count == 100:
        break
plainstring2[0]=inputTweets
# print type(plainstring2[0])
trainingDataFile = 'data/full_training_dataset.csv'
#classifierDumpFile = 'data/test/svm_test_model.pickle'
classifierDumpFile ='data/test/maxent_test_model.pickle'
keyword = 'iphone'
time = 'today'
trainingRequired = 0
print plainstring2
#sc = libsvm_classifier.SVMClassifier(plainstring2, keyword, time,\
 #                             trainingDataFile, classifierDumpFile, trainingRequired)


#sc = max_entropy_classifier.MaxEntClassifier(plainstring2, keyword, time,\
 #                                 trainingDataFile, classifierDumpFile, trainingRequired)


sc = naive_bayes_classifier.NaiveBayesClassifier(plainstring2, keyword, time,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)

sc.classify()
