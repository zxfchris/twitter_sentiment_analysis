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
    bc = baseline_classifier.BaselineClassifier(plainstring2, None, None)
    bc.classify()
    val = bc.getHTML()
elif(algorithm == 'naivebayes'):
    print "naive"
    trainingDataFile = 'data/training_neatfile.csv'
    classifierDumpFile = 'data/test/naivebayes_neat_model.pickle'
    trainingRequired = 1
    nb = naive_bayes_classifier.NaiveBayesClassifier(plainstring2, None, None,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
elif(algorithm == 'maxent'):
    trainingDataFile = 'data/training_neatfile.csv'
    classifierDumpFile = 'data/test/maxent_neat_model.pickle'
    print "maxent"
    trainingRequired = 1
    maxent = max_entropy_classifier.MaxEntClassifier(plainstring2, None, None,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
elif(algorithm == 'svm'):
    print "svm"
    trainingDataFile = 'data/training_neatfile.csv'
    classifierDumpFile = 'data/test/svm_neat_model.pickle'
    trainingRequired = 1
    sc = libsvm_classifier.SVMClassifier(plainstring2, None, None,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
    #sc.classify()
    # sc.accuracy()

print 'Done'
