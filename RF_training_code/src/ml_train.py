import numpy as np
# import sknn_jgd.mlp #uncomment if I want to run NN
import time
from sklearn.ensemble import RandomForestRegressor
from src.ml_io import write_netcdf_rf
from src.ml_io import write_netcdf_nn
import src.ml_load as ml_load
import pickle
import src.ml_plot as ml_plot
import os

# ---  build random forest or neural net  ---
def train_wrapper(f_ppi, o_ppi, training_expt, input_vert_dim, output_vert_dim,
                  input_vert_vars, output_vert_vars, flag_dict,
                  do_nn=False, n_iter=None, do_train=True, 
                  no_cos=True, use_rh=False,
                  max_z=40000.0,  
                  rain_only=False, 
                  n_trn_exs=None,  
                  plot_training_results=False, 
                  n_trees = 100, min_samples_leaf = 10, max_depth=25,
                  n_layers=2, n_hid_neur=10, n_stable=None, weight_decay=0.0, do_wind_input = False,do_diffusion=True,
                  scale_level=False,rewight_outputs = False, weight_list = [1,1],is_cheyenne=False,only_plot = False):
    """Loads training data and trains and stores estimator

    Args:
        f_ppi (dict): The type of preprocessing to do to the features (inputs)
        o_ppi (dict): The type of preprocessing to do to the targets (outputs)
        n_layers (int): Number of layers in the NN
        n_hid_neur (int): Number of hidden neurons in each layer
        n_iter (int): Number of iterations
        n_stable (int): Number of iterations after stability reached
        max_z (float): Don't train on data above this level
        weight_decay (float): Regularization strength. 0 is no regularization
        rain_only (bool): Only train on precipitating examples
        n_trn_exs (int): Number of training examples to learn on
        do_nn (bool): Use an ANN instead of a random forest 
        no_cos (bool): If true, don't weight by cosine(latitude)
        min_samples_leaf (int): minimum samples per leaf
        plot_training_results (bool): Whether to also plot the model on training data
        use_rh (bool): use generalized relative humidity instead of total non-precip water as feature
        do_train (bool): whether to train (just plot the results if false)
    Returns:
        str: String id of trained NN
    """
    # Load data (note LoadData seeds the random number generator)

    if not only_plot:
        datadir, trainfile, testfile, pp_str = ml_load.GetDataPath(training_expt, wind_input=do_wind_input,
                                                                   is_cheyenne=is_cheyenne)

        f, o, y, z, rho, p = ml_load.LoadData(trainfile, max_z, input_vert_vars=input_vert_vars, output_vert_vars=output_vert_vars,
                                              rain_only=rain_only, n_trn_exs=n_trn_exs, no_cos=no_cos, use_rh=use_rh,wind_input = do_wind_input,exclusion_flag=flag_dict['exclusion_flag'])

        #load test data
        tf, to, ty, tz, trho, tp = ml_load.LoadData(testfile, max_z, input_vert_vars=input_vert_vars, output_vert_vars=output_vert_vars,
                                                    rain_only=rain_only, n_trn_exs=n_trn_exs, no_cos=no_cos, use_rh=use_rh)



        # Scale data (both train and test)
        f_pp, f_scl, tf_scl, o_pp, o_scl, to_scl, pp_str = PreprocessData_tr_ts(f_ppi, f, tf, o_ppi, o, to, pp_str,
                                                                                n_trn_exs, z, input_vert_dim, input_vert_vars,
                                                                                output_vert_dim, output_vert_vars,scale_level,
                                                                                rewight_outputs=rewight_outputs,weight_list=weight_list) #Yani TO DO!!!



        #Scale test data
        # tf_pp, tf_scl, to_pp, to_scl, tpp_str = PreprocessData(f_ppi, tf, o_ppi, to, pp_str, n_trn_exs, tz)

        # # Scale data
        # f_pp, f_scl, o_pp, o_scl, pp_str = PreprocessData(f_ppi, f, o_ppi, o, pp_str, n_trn_exs, z)






        # Either build a random forest or build a neural netowrk
        if do_nn:
            regularize = CatchRegularization(weight_decay)
            est, est_str = BuildNN(max_z, n_layers, 'Rectifier', n_hid_neur,
                                    'momentum', pp_str, batch_size=100,
                                    n_stable=n_stable, n_iter=n_iter,
                                    learning_momentum=0.9, learning_rate=0.01,
                                    regularize=regularize,
                                    weight_decay=weight_decay,
                                    valid_size=0.2)
        else:
            est, est_str = BuildRandomForest(max_z, n_trees, min_samples_leaf,
                                             pp_str, max_depth, do_diffusion)

        est_str = UpdateName(no_cos, use_rh, rain_only, est_str)

        # Print details about the ML algorithm we are using
        print(est_str + ' Using ' + str(f.shape[0]) + ' training examples with ' +
              str(f.shape[1]) + ' input features and ' + str(o.shape[1]) +
              ' output targets')

        # Train the estimator
        if do_train:
          est, est_errors, train_score, test_score = train_est(est, est_str, f_scl, o_scl, tf_scl, to_scl, do_nn)

          est_str = est_str + 'te' + str(int(str(test_score)[2:4])) + '_tr' + str(int(str(train_score)[2:4]))

          # Save the estimator to access it later
          save_est(est, est_str, est_errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho, train_score, test_score,is_cheyenne)
        # Write a netcdf file for the gcm
        if do_nn:
         write_netcdf_nn(est_str, trainfile, rain_only, no_cos, use_rh,is_cheyenne)
        else:
             write_netcdf_rf(est_str, trainfile, output_vert_vars, output_vert_dim, rain_only, no_cos, use_rh,scale_level,
                             rewight_outputs=rewight_outputs,weight_list=weight_list,is_cheyenne=is_cheyenne)

        # Plot figures with testing data using all of it
    if only_plot:
        trainfile = '/glade/work/janniy/mldata/training_data/qobsTTFFFFFTF26TTTFTF48TFFFFFTFTFFF40FFTFTTF4848_training_x_no_subsampling.pkl'
        testfile = '/glade/work/janniy/mldata/training_data/qobsTTFFFFFTF26TTTFTF48TFFFFFTFTFFF40FFTFTTF4848_testing_x_no_subsampling.pkl'
        est_str = 'qobsTTFFFFFTF26TTTFTF48TFFFFFTFTFFF40FFTFTTF4848_F-NoSc_O-Stan_Ntr5000000_Nte972360_F_Tin_qin_qpin_latin_O_Tout_qout_qpout_RF_NTr10_MinS20max_d27_maxzinf_nocos_te50_tr54'

        trainfile = '/glade/work/janniy/mldata/training_data/qobsFFTFTFTFF0FFTFTF15FFFFFTFFFFTF815FFTFTTF00_training.pkl'
        testfile = '/glade/work/janniy/mldata/training_data/qobsFFTFTFTFF0FFTFTF15FFFFFTFFFFTF815FFTFTTF00_testing.pkl'
        est_str = 'qobsFFTFTFTFF0FFTFTF15FFFFFTFFFFTF815FFTFTTF00_F-NoSc_O-Stan_Ntr5000002_Nte972360_F_Tin_qin_uin_vinMinusSH_usurf_latin_O_tsurfCorr_qsurfCorr_tkz_RF_NTr10_MinS20max_d27_maxzinf_nocos_te70_tr75'

        trainfile = '/glade/scratch/janniy/mldata_tmp/training_data/qobsTTFFFFFTF26TTTFTF48TFFFFFTFTFFF320FFTFTTF4848_training_x_no_subsampling.pkl'
        testfile = '/glade/scratch/janniy/mldata_tmp/training_data/qobsTTFFFFFTF26TTTFTF48TFFFFFTFTFFF320FFTFTTF4848_testing_x_no_subsampling.pkl'
        est_str = 'qobsTTFFFFFTF26TTTFTF48TFFFFFTFTFFF320FFTFTTF4848_F-NoSc_O-Stan_Ntr1969020_Nte218790_F_Tin_qin_qpin_latin_O_Tout_qout_qpout_RF_NTr10_MinS7max_d27_maxzinf_nocos_te85_tr88'

    if only_plot:
        figpath = '/glade/scratch/janniy/figs_tmp_xy' + est_str + '/'
    else:
        figpath = './figs/' + est_str + '/'
    ml_plot.PlotAllFigs(est_str, testfile, do_nn, figpath, input_vert_vars, output_vert_vars,input_vert_dim,output_vert_dim, rain_only=rain_only,
                       n_trn_exs=n_trn_exs, no_cos=no_cos, use_rh=use_rh, wind_input = do_wind_input,scale_per_column=scale_level,
                        rewight_outputs=rewight_outputs,weight_list=weight_list,is_cheyenne=is_cheyenne)

    if plot_training_results: # note use n_trn_exs here as training data
        figpath = figpath + 'training_data/'
        ml_plot.PlotAllFigs(est_str, trainfile, do_nn, figpath, input_vert_vars, output_vert_vars,input_vert_dim,output_vert_dim,
                            rain_only=rain_only, n_trn_exs=n_trn_exs,
                            no_cos=no_cos, use_rh=use_rh, wind_input = do_wind_input,
                            rewight_outputs=rewight_outputs,weight_list=weight_list,is_cheyenne=is_cheyenne)
    return est_str



def PreprocessData_tr_ts(f_ppi, f, test_f, o_ppi, o, test_o, pp_str, n_trn_exs, z, input_vert_dim,
                         input_vert_vars, output_vert_dim, output_vert_vars,scale_per_column,
                         rewight_outputs=False,weight_list=[1,1]):
    """Transform data according to input preprocessor requirements and make
    make preprocessor string for saving"""

    #Preprocessing train features
    f_dict = ml_load.unpack_list(f, input_vert_vars, input_vert_dim)
    f_pp_dict = ml_load.init_pp_generalized(f_ppi,f_dict,input_vert_vars,scale_per_column)
    f_dict = ml_load.transform_data_generalized(f_ppi, f_pp_dict, f_dict, input_vert_vars, z,scale_per_column,rewight_outputs=False) #For random forest this is not necessary
    f = ml_load.pack_list(f_dict,input_vert_vars)

    #Preprocessing test features
    t_f_dict = ml_load.unpack_list(test_f, input_vert_vars, input_vert_dim,scale_per_column)
    t_f_dict = ml_load.transform_data_generalized(f_ppi, f_pp_dict, t_f_dict, input_vert_vars, z,scale_per_column,rewight_outputs=False) #For random forest this is not necessary
    t_f = ml_load.pack_list(t_f_dict, input_vert_vars)

    #Preprocessing train output
    o_dict = ml_load.unpack_list(o, output_vert_vars, output_vert_dim)
    o_pp_dict = ml_load.init_pp_generalized(o_ppi,o_dict,output_vert_vars,scale_per_column)
    o_dict = ml_load.transform_data_generalized(o_ppi, o_pp_dict, o_dict, output_vert_vars, z,scale_per_column,rewight_outputs=rewight_outputs,weight_list=weight_list) #For random forest this is not necessary
    o = ml_load.pack_list(o_dict, output_vert_vars)

    #Preprocessing test output
    t_o_dict = ml_load.unpack_list(test_o, output_vert_vars, output_vert_dim,scale_per_column)
    t_o_dict = ml_load.transform_data_generalized(o_ppi, o_pp_dict, t_o_dict, output_vert_vars, z,scale_per_column,
                                                  rewight_outputs=rewight_outputs,weight_list=weight_list) #For random forest this is not necessary
    t_o = ml_load.pack_list(t_o_dict, output_vert_vars)

    #output string
    pp_str =pp_str + 'F-' + f_ppi['name'][0:4] + '_'
    pp_str = pp_str + 'O-' + o_ppi['name'][0:4] + '_'
    # Add number of training examples to string
    pp_str = pp_str + 'Ntr' + str(f.shape[0]) + '_'
    pp_str = pp_str + 'Nte' + str(t_f.shape[0]) + '_'
    pp_str = pp_str + 'F_'
    for i in range(len(input_vert_dim)):
        pp_str = pp_str + input_vert_vars[i] + '_'
    pp_str = pp_str + 'O_'
    for i in range(len(output_vert_dim)):
        pp_str = pp_str  + output_vert_vars[i] + '_'


    return f_pp_dict,f, t_f, o_pp_dict,o, t_o, pp_str


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

def CatchRegularization(weight_decay):
    """scikit-neuralnetwork seems to have a bug if regularization is set to zero"""
    if weight_decay > 0.0:
        regularize = 'L2'
    else:
        regularize = None
    return regularize

def UpdateName(no_cos, use_rh, rain_only, est_str):
    if no_cos:
        est_str = est_str + '_nocos_'
    if use_rh:
        est_str = est_str + '_rh'
    if rain_only:
        est_str = est_str + '_rain'

    return est_str

def save_est(est, est_str, est_errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho, train_score, test_score,is_cheyenne=False):
    """Save estimator"""
    if is_cheyenne == False:  # On aimsir/esker
        base_dir = '/net/aimsir/archive1/janniy/'
    else:
        base_dir = '/glade/scratch/janniy/'

    if not os.path.exists(base_dir + 'mldata_tmp/regressors/'):
        os.makedirs(base_dir + 'mldata_tmp/regressors/')
    pickle.dump([est, est_str, est_errors, f_ppi, o_ppi, f_pp, o_pp, y, z, p, rho], open(base_dir + 'mldata_tmp/regressors/' + est_str + '.pkl', 'wb'))

def store_stats(i, avg_train_error, best_train_error, avg_valid_error,
                best_valid_error, avg_train_obj_error, best_train_obj_error,
                **_):
    if i == 1:
        global errors_stored
        errors_stored = []
    errors_stored.append((avg_train_error, best_train_error,
                          avg_valid_error, best_valid_error,
                          avg_train_obj_error, best_train_obj_error))


def BuildNN(max_z, n_layers, actv_fnc, hid_neur, learning_rule, pp_str,
             batch_size=100, n_iter=None, n_stable=None,
             learning_rate=0.01, learning_momentum=0.9,
             regularize='L2', weight_decay=0.0, valid_size=0.5,
             f_stable=.001):
    """Builds a multi-layer perceptron via the scikit neural network interface
       Note using lasagne and theano
    """
    # First build layers
    actv_fnc = n_layers*[actv_fnc]
    hid_neur = n_layers*[hid_neur]
    layers = [sknn_jgd.mlp.Layer(f, units=h) for f, h in zip(actv_fnc,
                                                             hid_neur)]
    # Append a linear output layer 
    layers.append(sknn_jgd.mlp.Layer("Linear"))
    est = sknn_jgd.mlp.Regressor(layers,
                                 n_iter=n_iter,
                                 batch_size=batch_size,
                                 learning_rule=learning_rule,
                                 learning_rate=learning_rate,
                                 learning_momentum=learning_momentum,
                                 regularize=regularize,
                                 weight_decay=weight_decay,
                                 n_stable=n_stable,
                                 valid_size=valid_size,
                                 f_stable=f_stable,
                                 callback={'on_epoch_finish': store_stats})
    # Write nn string
    layerstr = '_'.join([str(h) + f[0] for h, f in zip(hid_neur, actv_fnc)])
    if learning_rule == 'momentum':
        lrn_str = str(learning_momentum)
    else:
        lrn_str = str(learning_rate)
    # Construct name
    est_str = pp_str + "r_" + layerstr + "_" + learning_rule[0:3] +\
        lrn_str
    # If using regularization, add that to the name too
    if weight_decay > 0.0:
        est_str = est_str + 'reg' + str(weight_decay)
    # Add the number of iterations too
    est_str = est_str + '_Niter' + str(n_iter)
    # Add the maximum level
    est_str = est_str + '_maxz' + str(max_z)
    return est, est_str


def BuildRandomForest(max_z, n_trees, min_samples_leaf, pp_str, max_depth, do_diffusion):

    est = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf = min_samples_leaf, max_features = 1.0/3.0, max_depth=max_depth, n_jobs=10, random_state = 123)
    est_str = pp_str + 'RF_NTr' + str(n_trees) + '_MinS' + str(min_samples_leaf) + 'max_d' + str(max_depth)  + '_maxz' + str(max_z)

    return est, est_str


def train_est(est, est_str, f_scl, o_scl, tf_scl, to_scl, do_nn):
    """Train estimator"""

    # Initialize
    start = time.time()

    # Train the model using training data
    est.fit(f_scl, o_scl)
    train_int_score = max(int(np.ceil(f_scl.shape[0] / 10)), 10000)
    train_score = est.score(f_scl[0:train_int_score, :], o_scl[0:train_int_score, :])
    test_score = est.score(tf_scl, to_scl)

    end = time.time()
    print("Training Score: {:.4f} for Model {:s} ({:.1f} seconds)".format(
                                              train_score, est_str, end-start))
    print("Test  Score: {:.4f} for Model {:s} ({:.1f} seconds)".format(
                                              test_score, est_str, end - start))
    if do_nn:
     # This is an n_iter x 4 array...see score_stats
     errors = np.asarray(errors_stored)
    else:
     errors = np.empty(0)

    # Return the fitted models and the scores
    return est, errors, train_score, test_score


