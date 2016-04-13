import numpy
import os
from sklearn.datasets import load_svmlight_file


def convert_sentiment(sentiment):
    return {
        2:"positive",
        0:"negative",
        1:"neutral",
    }.get(sentiment,"neutral")


def main():
    model_dir = 'models/model3_2016devtest_opt'
    pred_files = [
        'save_predictions_Train.npy',
        'save_predictions_train_2016.npy',
        'save_predictions_dev2016.npy',
        'save_predictions_devtest2016.npy',
        'save_predictions_2014.npy',
        'save_predictions_2015.npy',
        'save_predictions_dev.npy',
        'save_predictions_test_2016.npy'
    ]

    feature_files = [
        'featuresData.csv',
        'featuresTrain2016.csv',
        'featuresDev2016.csv',
        'featuresDevTest2016.csv',
        'featuresTest2014.csv',
        'featuresTest2015.csv',
        'featuresDevSet.csv',
        'featuresTest2016.csv'
    ]

    pred_files = map(lambda x: os.path.join(model_dir,x),pred_files)
    feature_files = map(lambda x: os.path.join(model_dir,x),feature_files)

    data_dir = 'semeval'
    data_files = [
        ('task-B-train-plus-dev.tsv',4),
        ('task-A-train-2016.tsv',3),
        ('task-A-dev-2016.tsv',3),
        ('task-A-devtest-2016.tsv',3),
        ('task-B-test2014-twitter.tsv',4),
        ('task-B-test2015-twitter.tsv',4),
        ('twitter-test-gold-B.downloaded.tsv',4),
        ('SemEval2016-task4-test.subtask-A_pred.txt',3)
    ]

    names = [
        'train_old',
        'train_2016',
        'dev-2016',
        'devtest-2016',
        'test-2014',
        'test-2015',
        'dev-2013-test',
        'test-2016'
    ]

    data_files = map(lambda x: (os.path.join(data_dir,x[0]),x[1]),data_files)

    outpath = os.path.join(model_dir,'id_pred_lab_tweet_feat')
    outfile = os.path.join(outpath,'all_tweets_file.txt')
    output_all = open(outfile,'w')
    for (dfile,ffile,pfile,name) in zip(data_files,feature_files,pred_files,names):
        outfile = os.path.join(outpath,'all_tweets_file_{}.txt'.format(name))
        output = open(outfile,'w')
        data_file = open(dfile[0],'r')
        feature_file = open(ffile,'r')
        n_cols = dfile[1]

        data = data_file.readlines()
        features = feature_file.readlines()
        predictions = numpy.load(pfile)
        assert  len(data) == len(features) and len(features) == predictions.shape[0]

        for d_line,f_line,p_line in zip(data,features,predictions):
            data_line = d_line.replace('\n','').split('\t')
            t_id = data_line[0]
            pred = convert_sentiment(p_line)
            label = data_line[n_cols-2]
            tweet = data_line[n_cols-1]

            sentence_vec = ' '.join(f_line.replace('\n','').split(' ')[1:])
            outline = '\t'.join([t_id,pred,label,tweet,sentence_vec])
            output.write(outline+'\n')
            output_all.write(outline+'\n')

if __name__ == '__main__':
    main()