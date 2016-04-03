#import regex
import re
import csv
import pprint
import nltk.classify
import libsvm_classifier

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

# testTweet = "@PrincessSuperC Hey Cici sweetheart! Just wanted to let u know I luv u! OH! and will the mixtape drop soon? FANTASY RIDE MAY 5TH!!!!"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "@Msdebramaye I heard about that contest! Congrats girl!!"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "UNC!!! NCAA Champs!! Franklin St.: I WAS THERE!! WILD AND CRAZY!!!!!! Nothing like it...EVER http://tinyurl.com/49955t3"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "Do you Share More #jokes #quotes #music #photos or #news #articles on #Facebook or #Twitter?"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "Good night #Twitter and #TheLegionoftheFallen.  5:45am cimes awfully early!"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "I just finished a 2.66 mi run with a pace of 11'14\"/mi with Nike+ GPS. #nikeplus #makeitcount"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "Disappointing day. Attended a car boot sale to raise some funds for the sanctuary, made a total of 88p after the entry fee - sigh"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "no more taking Irish car bombs with strange Australian women who can drink like rockstars...my head hurts."
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "Just had some bloodwork done. My arm hurts"
# inputTweets.append(unicode(testTweet, "utf-8"))
# testTweet = "I hate you"
# inputTweets.append(unicode(testTweet, "utf-8"))

movieComments = open('/Users/zhengxifeng/Downloads/Maleficent_update', 'rb')
count = 0
for comment in movieComments:
    # print comment
    inputTweets.append(unicode(comment, "utf-8"))
    count = count + 1
#     if count == 1000:
#         break
plainstring2[0]=inputTweets
# print type(plainstring2[0])
trainingDataFile = 'data/full_training_dataset.csv'           
classifierDumpFile = 'data/test/svm_test_model.pickle'
keyword = 'iphone'
time = 'today'
trainingRequired = 0
print plainstring2
sc = libsvm_classifier.SVMClassifier(plainstring2, keyword, time,\
                              trainingDataFile, classifierDumpFile, trainingRequired)
sc.classify()