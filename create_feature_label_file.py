import os
import numpy

def main():
    model_dir = 'models/model3_2016devtest_opt'
    pred_files_train = [
        'save_predictions_Train.npy',
        'save_predictions_train_2016.npy',
        'save_predictions_dev2016.npy',
        'save_predictions_devtest2016.npy',

    ]

    pred_files_test = [
        'save_predictions_2014.npy',
        'save_predictions_2015.npy',
        'save_predictions_dev.npy',
        'save_predictions_test_2016.npy'
    ]

    feature_files_train = [
        'featuresData.csv',
        'featuresTrain2016.csv',
        'featuresDev2016.csv',
        'featuresDevTest2016.csv'
    ]

    feature_files_test = [
        'featuresTest2014.csv',
        'featuresTest2015.csv',
        'featuresDevSet.csv',
        'featuresTest2016.csv'
    ]

    test_names = [
        'test_2014',
        'test_2015',
        'test2013-dev',
        'test2016'
    ]

    pred_files = map(lambda x: os.path.join(model_dir,x),pred_files_train)
    feature_files = map(lambda x: os.path.join(model_dir,x),feature_files_train)
    outpaht = os.path.join(model_dir,'lab_feat_pred')
    outfile = os.path.join(outpaht,'train_features_pred.txt')
    output = open(outfile,'w')
    for (ffile,pfile) in zip(feature_files,pred_files):
        feature_file = open(ffile,'r')
        features = feature_file.readlines()
        predictions = numpy.load(pfile)
        assert  len(features) == predictions.shape[0]

        for f_line,p_line in zip(features,predictions):
            outline = ' '.join([f_line.replace('\n',''),str(p_line)])
            output.write(outline + '\n')


    pred_files = map(lambda x: os.path.join(model_dir,x),pred_files_test)
    feature_files = map(lambda x: os.path.join(model_dir,x),feature_files_test)


    for (ffile,pfile,name) in zip(feature_files,pred_files,test_names):
        outfile = os.path.join(outpaht,'{}-features_pred.txt'.format(name))
        output = open(outfile,'w')

        feature_file = open(ffile,'r')
        features = feature_file.readlines()
        predictions = numpy.load(pfile)
        assert  len(features) == predictions.shape[0]

        for f_line,p_line in zip(features,predictions):
            outline = ' '.join([f_line.replace('\n',''),str(p_line)])
            output.write(outline + '\n')

if __name__ == '__main__':
    main()