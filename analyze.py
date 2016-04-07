import get_twitter_data
import baseline_classifier,naive_bayes_classifier,max_entropy_classifier
import libsvm_classifier
import json,sys,pickle


keyword = 'worst'
time = 'today'
twitterData = get_twitter_data.TwitterData()
tweets = twitterData.getTwitterData(keyword, time)

#algorithm = 'baseline'
#algorithm = 'naivebayes'
#algorithm = 'maxent'
#algorithm = 'svm'

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
    trainingDataFile = 'data/training_neatfile.csv'

    classifierDumpFile = 'data/test/naivebayes_neat_model.pickle'
    trainingRequired = 1
    nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
    nb.classify()
elif(algorithm == 'maxent'):
    trainingDataFile = 'data/training_neatfile.csv'
    classifierDumpFile = 'data/test/maxent_neat_model.pickle'
    print "maxent"
    trainingRequired = 0
    print "Start",time
    maxent = max_entropy_classifier.MaxEntClassifier(tweets, keyword, time,\
                                  trainingDataFile, classifierDumpFile, trainingRequired)
    print "End ",time
    maxent.analyzeTweets()
    maxent.classify()
    print "Next End ",time

elif(algorithm == 'svm'):
    #trainingDataFile = 'data/training_trimmed.csv'
    #trainingDataFile = 'data/full_training_dataset.csv'
    print "svm"
    testDataFile = 'data/test/test.csv'
    trainingDataFile = 'data/training_neatfile.csv'
    classifierDumpFile = 'data/test/svm_neat_model.pickle'
    trainingRequired = 0
    # sc = libsvm_classifier.SVMClassifier(tweets, keyword, time,\
    #                               trainingDataFile, classifierDumpFile, trainingRequired)
    sc = libsvm_classifier.SVMClassifier(tweets, keyword, time,\
                                  testDataFile, classifierDumpFile, trainingRequired)
    sc.classify()

print 'Done'
