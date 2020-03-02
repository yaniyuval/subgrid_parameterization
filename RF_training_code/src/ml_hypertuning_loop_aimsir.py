import numpy as np
#import sknn_jgd.mlp
import time
from sklearn.ensemble import RandomForestRegressor
from src.ml_io import write_netcdf_rf
from src.ml_io import write_netcdf_nn
import src.ml_load as ml_load
import pickle
#import src.ml_plot as ml_plot
import os
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
# from sklearn.decomposition import PCA



#from xgboost.sklearn import XGBRegressor

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / abs(test_labels/2 + predictions/2))
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy





def PreprocessData(f_ppi, f, o_ppi, o, pp_str, n_trn_exs, z):
    """Transform data according to input preprocessor requirements and make
    make preprocessor string for saving"""
    f_pp = ml_load.init_pp(f_ppi, f)
    f = ml_load.transform_data(f_ppi, f_pp, f, z)
    o_pp = ml_load.init_pp(o_ppi, o)
    o = ml_load.transform_data(o_ppi, o_pp, o, z)
    # Make preprocessor string for saving
    pp_str = pp_str + 'F-' + f_ppi['name'] + '_'
    pp_str = pp_str + 'O-' + o_ppi['name'] + '_'
    # Add number of training examples to string
    pp_str = pp_str + 'Ntrnex' + str(n_trn_exs) + '_'
    return f_pp, f, o_pp, o, pp_str



f_ppi = {'name': 'NoScaler'}
o_ppi = {'name': 'SimpleO'}

do_train = True
#max_z=20000
max_z=np.Inf
min_samples_leaf = 10
n_trees = 11
n_trn_exs = 10000000000
use_rh = False
no_cos = True

training_expt = 'qobs'
#training_expt2 = 'qobs4K
rain_only = False


datadir, trainfile, testfile, pp_str = ml_load.GetDataPath(training_expt)


#Train data
f, o, y, z, rho, p = ml_load.LoadData(trainfile, max_z, rain_only=rain_only, n_trn_exs=n_trn_exs, no_cos=no_cos, use_rh=use_rh)
f_pp, f_scl, o_pp, o_scl, pp_str = PreprocessData(f_ppi, f, o_ppi, o, pp_str, n_trn_exs, z)
print('read training data')
#test data
tf, to, ty, tz, trho, tp = ml_load.LoadData(testfile, max_z, rain_only=rain_only, n_trn_exs=n_trn_exs, no_cos=no_cos, use_rh=use_rh)
tf_pp, tf_scl, to_pp, to_scl, tpp_str = PreprocessData(f_ppi, tf, o_ppi, to, pp_str, n_trn_exs, tz)
print('read test data')


#X_train = np.concatenate((f_scl,tf_scl))
#Y_train = np.concatenate((o_scl,to_scl))
#per_test = 0.1
#per_train = 1 - per_test
#max_ind = np.int(np.ceil(X_train.shape[0]*per_train))
#f_scl = X_train[0:max_ind,:]
#tf_scl = X_train[max_ind:,:]
#o_scl = Y_train[0:max_ind,:]
#to_scl = Y_train[max_ind:,:]


n_estimators = [10,15]
max_features = ['auto']
max_depth = [10,20,40,50]
# Minimum number of samples required to split a node
min_samples_split = [2,30]
# Minimum number of samples required at each leaf node

max_features = ['auto','sqrt']

min_samples_leaf = [10]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
file = open('res_RF_time_w_diffusion.txt','w')
# num_train = 100#f_scl.shape[0]
# print("the number of training samples are",num_train)
# f_scl = f_scl[0:num_train]
# o_scl = o_scl[0:num_train]
# tf_scl = tf_scl[0:num_train]
# to_scl = to_scl[0:num_train]



for i in n_estimators:
    print(i)
    n_estimators_tmp = i
    for j in max_depth:
        max_depth_tmp = j
        for k in min_samples_leaf:
            min_samples_leaf_tmp = k
            for l in max_features:
                max_features_tmp = l
                for m in min_samples_split:
                    min_samples_split_tmp = m
                    file.write("max_features : " + str(max_features_tmp)  + "\n n estimators : " + str(n_estimators_tmp)+  "\n max_depth: " +  str(max_depth_tmp) + "\n min_samples_leaf: " + str(min_samples_leaf_tmp)+ "\n min_samples_split: " +str(min_samples_split_tmp)   + '\n')
                    print("n estimators: ", n_estimators_tmp, "max_depth", max_depth_tmp, "min_samples_leaf", min_samples_leaf_tmp, "max_features: " , max_features_tmp, "min_samples_split: ", min_samples_split_tmp)
                    start = time.time()
                    rf = RandomForestRegressor(max_depth = max_depth_tmp, n_estimators = n_estimators_tmp,n_jobs = 5, max_features=max_features_tmp, min_samples_split=min_samples_split_tmp, min_samples_leaf=min_samples_leaf_tmp)
                    rf.fit(f_scl, o_scl)
                    # scores = cross_val_score(estimator=rf,
                    #                 X=f_scl,
                    #                y=o_scl,
                    #               cv=3,   ## This is the split size of the train data (should be larger for smaller data - to reduce bias)
                    #              n_jobs=1)
                    # print('CV accuracy scores: %s' % scores)
                    # print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
                    # for item in scores:
                    #     file.write("scores: %s\n" %item)
                    #file.write('CV accuracy: %.3f +/- %.3f\n' % (np.mean(scores), np.std(scores)))

                    train_score = rf.score(f_scl, o_scl)
                    test_score = rf.score(tf_scl, to_scl)
                    start_score = time.time()
                    rf.predict(tf_scl)
                    end_score = time.time()
                    print("train score:", train_score, "test score: ", test_score)
                    file.write('train_score : %.3f\n' % (train_score))
                    file.write('test_score : %.3f\n' % (test_score))
                    file.write("time to test the model: %.3f\n" % (end_score-start_score))
                    end = time.time()
                    print("time to run the model ({:.1f} seconds)".format( end-start))
                    file.write('time of run : %.3f\n' % (end- start))


file.close()


