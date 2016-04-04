import get_twitter_data
import baseline_classifier,naive_bayes_classifier,max_entropy_classifier
import libsvm_classifier
import json,sys,pickle

plainstring2 = dict()
inputTweets = list()
inputTweets.append(unicode('', "utf-8"))
plainstring2[0]=inputTweets

if(len(sys.argv) < 2):
    print "Please choose the algorithm to test, sytanx = python analyze.py (svm|naivebayes|maxent)"
    exit()
    
algorithm = sys.argv[1]

if(algorithm == 'baseline'):
    bc = baseline_classifier.BaselineClassifier(tweets, keyword, time)
    bc.classify()
    val = bc.getHTML()
elif(algorithm == 'naivebayes'):
    print "naive"
    testDataFile = 'data/test/test.csv'
    classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
    print "naivebaes"
    trainingRequired = 0
    nb = naive_bayes_classifier.NaiveBayesClassifier(plainstring2, keyword, time,\
                                  testDataFile, classifierDumpFile, trainingRequired)
    #nb.classify()
    nb.accuracy()
elif(algorithm == 'maxent'):
    testDataFile = 'data/test/test.csv'
    classifierDumpFile = 'data/test/maxent_test_model.pickle'
    print "maxent"
    trainingRequired = 0
    print "Start",time
    maxent = max_entropy_classifier.MaxEntClassifier(plainstring2, keyword, time,\
                                  testDataFile, classifierDumpFile, trainingRequired)
    print "End ",time
    maxent.analyzeTweets()
    #maxent.classify()
    print "Next End ",time
    maxent.accuracy()
elif(algorithm == 'svm'):
    print "svm"
    testDataFile = 'data/test/test.csv'
    classifierDumpFile = 'data/test/svm_test_model.pickle'
    trainingRequired = 0
    print tweets
    sc = libsvm_classifier.SVMClassifier(plainstring2, keyword, time,\
                                  testDataFile, classifierDumpFile, trainingRequired)
    #sc.classify()
    sc.accuracy()

print 'Done'
