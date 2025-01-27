# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

from scipy.stats import pearsonr


# importing keras
import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

# importing util libraries
import datetime
import time
import math
import os
import importlib

# importing custom library
import DNN_models
import exception_handle

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


#fix np.random.seed for reproducibility in numpy processing
np.random.seed(7)


class HostBlockedKFold(BaseCrossValidator):
    """
    A cross-validator that provides train/test indices to split data into train/test sets,
    ensuring that samples from the same host are not split between training and test sets.

    Parameters:
    -----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    host_to_indices_dict : dict, default=None
        A dictionary mapping each host to a list of indices corresponding to that host's samples.
    shuffle : boolean, default=False
        Whether to shuffle the hosts before splitting into batches.
    random_state : int, RandomState instance or None, default=None
        When shuffle is True, random_state affects the order of the hosts.
        Pass an int for reproducible output across multiple function calls.
    """
    def __init__(self, n_splits=5, host_to_indices_dict=None, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.host_to_indices_dict = host_to_indices_dict
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. This parameter is not used, but is included for
            compatibility with other cross-validators.

        Yields:
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        hosts = list(self.host_to_indices_dict.keys())
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(hosts)

        host_splits = np.array_split(hosts, self.n_splits)
        
        for i in range(self.n_splits):
            test_hosts = host_splits[i]
            train_hosts = [host for split in host_splits[:i] + host_splits[i+1:] for host in split]
            
            test_indices = [idx for host in test_hosts for idx in self.host_to_indices_dict[host]]
            train_indices = [idx for host in train_hosts for idx in self.host_to_indices_dict[host]]
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator

        Parameters:
        -----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns:
        --------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

class DeepMicrobiome(object):
    """
    A class for handling microbiome data, including loading, preprocessing, and representation learning.

    Attributes:
        t_start (float): Start time of the process.
        filename (str): Name of the input file.
        data (str): Name of the dataset.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory containing the data.
        prefix (str): Prefix for output files.
        representation_only (bool): Flag to indicate if only representation learning is performed.
        additional_filenames (list): List of additional data filenames.
        additional_X (list): List of additional feature matrices.
        additional_y (list): List of additional label vectors.
        pretrain_X (numpy.ndarray, optional): Pretraining feature matrix.
        pretrain_Y (numpy.ndarray, optional): Pretraining label vector.
        use_all_data_for_train (bool): Flag to use all data for training (meaning an additional dataset should be used for testing).
        train_host_to_indices_dict (dict, optional): Mapping of hosts to indices for training data.
        test_host_to_indices_dict (dict, optional): Mapping of hosts to indices for test data.
        host_to_indices_dict (dict, optional): Mapping of hosts to indices for all data.
    """
    def __init__(self, data, seed, data_dir, host_to_indices_dict_str=None):
        """
        Initialize the DeepMicrobiome object.

        Args:
            data (str): Name of the input file.
            seed (int): Random seed for reproducibility.
            data_dir (str): Directory containing the data.
            host_to_indices_dict_str (str, optional): Filename of the host to indices dictionary.
        """
        self.t_start = time.time()
        self.filename = str(data)
        self.data = self.filename.split('.')[0]
        self.seed = seed
        self.data_dir = data_dir
        self.prefix = ''
        self.representation_only = False

        self.additional_filenames = []
        self.additional_X = []
        self.additional_y = []

        self.pretrain_X = None
        self.pretrain_Y = None

        self.use_all_data_for_train = False
        self.train_host_to_indices_dict = None
        self.test_host_to_indices_dict = None

        if host_to_indices_dict_str is not None:
            self.loadHostToIndicesDict(host_to_indices_dict_str)
        else:
            self.host_to_indices_dict = None

    def loadData(self, feature_string, label_string, label_dict, dtype=None):
        """
        Load data from a file and split it into training and test sets.

        Args:
            feature_string (str): String to identify feature rows in the input file.
            label_string (str): String to identify the label row in the input file.
            label_dict (dict): Dictionary to map label strings to integers.
            dtype (numpy.dtype, optional): Data type for the feature matrix.
        """
        print("Loading data")
        # read file
        filename = self.data_dir + "data/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # select rows having feature index identifier string
        X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T

        # get class labels
        Y = raw.loc[label_string] #'disease'
        Y = Y.replace(label_dict)

        if self.host_to_indices_dict is None:
            # train and test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test, self.train_host_to_indices_dict, self.test_host_to_indices_dict = self.split_data_by_host(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values, host_to_indices_dict=self.host_to_indices_dict)

        self.printDataShapes()
    
    def loadAdditionalData(self, additional_data, additional_labels, dtype=None):
        """
        Load additional datasets and their corresponding labels.

        Args:
            additional_data (list): List of filenames for additional datasets.
            additional_labels (list): List of filenames for additional labels.
            dtype (numpy.dtype, optional): Data type for the feature matrices.
        """
        print("Loading additional data")
        self.additional_filenames = additional_data
        # read files
        for fn, ln in zip(additional_data, additional_labels):
            filename = self.data_dir + "data/" + fn
            label_filename = self.data_dir + "data/" + ln
            if os.path.isfile(filename) and os.path.isfile(label_filename):
                raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
                label = pd.read_csv(label_filename, sep=',', index_col=False, header=None)
            else:
                if not os.path.isfile(filename):
                    print("FileNotFoundError: File {} does not exist".format(filename))
                if not os.path.isfile(label_filename):
                    print("FileNotFoundError: File {} does not exist".format(label_filename))
                exit()

            # label data validity check
            if not label.values.shape[1] > 1:
                label_flatten = label.values.reshape((label.values.shape[0]))
            else:
                print('FileSpecificationError: The label file contains more than 1 column.')
                exit()

            # train and test split
            X_train, X_test, y_train, y_test = train_test_split(raw.values.astype(dtype),
                                                                label_flatten.astype('int'), test_size=0.2,
                                                                random_state=self.seed,
                                                                stratify=label_flatten)
            X = np.concatenate((X_train, X_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)
            self.additional_X.append(X)
            self.additional_y.append(y)
        self.printDataShapes()

    def loadCustomData(self, dtype=None):
        """
        Load custom data without labels.

        Args:
            dtype (numpy.dtype, optional): Data type for the feature matrix.
        """
        # read file
        filename = self.data_dir + "data/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()
        print("Loading custom data without labels")

        # load data
        self.X_train = raw.values.astype(dtype)

        # put nothing or zeros for y_train, y_test, and X_test
        self.y_train = np.zeros(shape=(self.X_train.shape[0])).astype(dtype)
        self.X_test = np.zeros(shape=(1,self.X_train.shape[1])).astype(dtype)
        self.y_test = np.zeros(shape=(1,)).astype(dtype)
        self.printDataShapes(train_only=True)

    def loadCustomDataWithLabels(self, label_data, dtype=None, use_all_data_for_train=False):
        """
        Load custom data with labels and optionally split into train and test sets.
        Is the primary function for loading data.

        Args:
            label_data (str): Filename of the label data.
            dtype (numpy.dtype, optional): Data type for the feature matrix.
            use_all_data_for_train (bool, optional): If True, use all data for training. Should only be used if we want to use an additional dataset for testing.
        """
        # read file
        filename = self.data_dir + "data/" + self.filename
        label_filename = self.data_dir + "data/" + label_data
        if os.path.isfile(filename) and os.path.isfile(label_filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
            label = pd.read_csv(label_filename, sep=',', index_col=False, header=None)
        else:
            if not os.path.isfile(filename):
                print("FileNotFoundError: File {} does not exist".format(filename))
            if not os.path.isfile(label_filename):
                print("FileNotFoundError: File {} does not exist".format(label_filename))
            exit()

        # label data validity check
        if not label.values.shape[1] > 1:
            label_flatten = label.values.reshape((label.values.shape[0]))
        else:
            print('FileSpecificationError: The label file contains more than 1 column.')
            exit()

        # train and test split
        print("Loading custom data with labels")
        indices = np.arange(raw.shape[0])
        if self.host_to_indices_dict is not None:
            # split data by host
            self.train_indices, self.test_indices, self.X_train, self.X_test, self.y_train, self.y_test, self.train_host_to_indices_dict, self.test_host_to_indices_dict = self.split_data_by_host(raw.values.astype(dtype),
                                                                                                                           label_flatten,
                                                                                                                           return_indices=True,
                                                                                                                           test_size=0.2, 
                                                                                                                           random_state=self.seed, 
                                                                                                                           stratify=label_flatten,
                                                                                                                           host_to_indices_dict=self.host_to_indices_dict)
            self.train_indices = np.array(self.train_indices)
            self.test_indices = np.array(self.test_indices)
            self.prefix += "host_split"
        else:
            self.train_indices, self.test_indices, self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(indices, 
                                                                                                                           raw.values.astype(dtype),
                                                                                                                           label_flatten,
                                                                                                                           test_size=0.2, 
                                                                                                                           random_state=self.seed, 
                                                                                                                           stratify=label_flatten)
        if use_all_data_for_train:
            # Combine train and test data
            self.use_all_data_for_train = True
            self.X_train = np.concatenate((self.X_train, self.X_test), axis=0)
            self.y_train = np.concatenate((self.y_train, self.y_test), axis=0)
            self.train_indices = np.concatenate((self.train_indices, self.test_indices), axis=0)
            
            # Adjust prefix to indicate all data used for training
            self.prefix += "_all_data_train"
            
            print("All data used for training. Test results should not be trusted.")
        print("Train indices: ", self.train_indices)
        print("Test indices: ", self.test_indices)
        print("Train labels: ", self.y_train)
        print("Test labels: ", self.y_test)
        print("Train data: ", self.X_train)
        print("Test data: ", self.X_test)
        # We can optionally include unlabeled data for pretraining in the custom data with labels of -100
        # However, this will cause issues with the splitting of data by host, so if we need to split by host, 
        # then we should use the loadPretrainingData function instead.
        if -100 in self.y_test:
            # Identify indices where test labels are -100
            test_indices_to_move = np.where(self.y_test == -100)[0]
            print("Test indices to move: ", test_indices_to_move)
            train_indices_to_add = self.test_indices[test_indices_to_move]

            # Move the identified test instances to train
            self.X_train = np.concatenate((self.X_train, self.X_test[test_indices_to_move]), axis=0)
            self.y_train = np.concatenate((self.y_train, self.y_test[test_indices_to_move]), axis=0)
            self.train_indices = np.concatenate((self.train_indices, train_indices_to_add), axis=0)

            # Remove the moved instances from test
            self.X_test = np.delete(self.X_test, test_indices_to_move, axis=0)
            self.y_test = np.delete(self.y_test, test_indices_to_move, axis=0)
            self.test_indices = np.delete(self.test_indices, test_indices_to_move, axis=0)

        self.printDataShapes()

    def loadPretrainingData(self, pretrain_data, dtype=None):
        """
        Load pretraining data.

        Args:
            pretrain_data (str): Filename of the pretraining data.
            dtype (numpy.dtype, optional): Data type for the feature matrix.
        """
        print("Loading pretraining data")
        filename = self.data_dir + "data/" + pretrain_data
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        self.pretrain_X = raw.values.astype(dtype)
        self.pretrain_Y = np.ones(raw.shape[0])
        frac_positive = (self.y_train[self.y_train != -100]).sum() / (self.y_train[self.y_train != -100]).shape[0]
        self.pretrain_Y = np.random.choice([0, 1], size=self.pretrain_Y.shape[0], p=[1-frac_positive, frac_positive])
        self.printDataShapes()
    
    def loadHostToIndicesDict(self, host_to_indices_dict_str):
        """
        Load the host to indices dictionary from a pickle file.

        Args:
            host_to_indices_dict_str (str): Filename of the host to indices dictionary (without extension).
        """
        print("Loading host to indices dictionary")
        file_path = self.data_dir + "data/host_to_indices/" + host_to_indices_dict_str + ".pkl"
        if os.path.isfile(file_path):
            self.host_to_indices_dict = pickle.load(open(file_path, 'rb'))
            print(f"Loaded host to indices dictionary from {file_path}")
        else:
            print("FileNotFoundError: File {} does not exist".format(file_path))
            exit()

    #Principal Component Analysis
    def pca(self, ratio=0.99):
        print("Performing PCA")
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'PCA_'

        # PCA
        pca = PCA()
        pca.fit(self.X_train if self.pretrain_X is None else self.pretrain_X)
        n_comp = 0
        ratio_sum = 0.0

        for comp in pca.explained_variance_ratio_:
            ratio_sum += comp
            n_comp += 1
            if ratio_sum >= ratio:  # Selecting components explaining 99% of variance
                break

        pca = PCA(n_components=n_comp)
        pca.fit(self.X_train if self.pretrain_X is None else self.pretrain_X)

        X_train = pca.transform(self.X_train)
        if self.pretrain_X is not None and self.pretrain_X.shape[0] > 0:
            pretrain_X = pca.transform(self.pretrain_X)
            self.pretrain_X = pretrain_X
        X_test = pca.transform(self.X_test)

        # applying the eigenvectors to the whole training and the test set.
        self.X_train = X_train
        self.X_test = X_test
       

        # applying the eigenvectors to the additional datasets:
        for i in range(len(self.additional_filenames)):
            self.additional_X[i] = pca.transform(self.additional_X[i])
        self.printDataShapes()

    #Gausian Random Projection
    def rp(self):
        print("Performing Gaussian Random Projection")
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'RandP_'
        # GRP
        rf = GaussianRandomProjection(eps=0.5)
        rf.fit(self.X_train if self.pretrain_X is None else self.pretrain_X)

        # applying GRP to the whole training and the test set.
        self.X_train = rf.transform(self.X_train)
        self.X_test = rf.transform(self.X_test)
        self.pretrain_X = rf.transform(self.pretrain_X)

        # applying GRP to the additional datasets:
        for i in range(len(self.additional_filenames)):
            self.additional_X[i] = rf.transform(self.additional_X[i])
        self.printDataShapes()

    #Shallow Autoencoder & Deep Autoencoder
    def ae(self, dims = [50], epochs= 2000, batch_size=100, verbose=2, loss='mean_squared_error', latent_act=False, output_act=False, act='relu', patience=20, val_rate=0.2, no_trn=False, lr=0.0005):
        print("Training shallow autoencoder")
        # manipulating an experiment identifier in the output file
        if patience != 20:
            self.prefix += 'p' + str(patience) + '_'
        if len(dims) == 1:
            self.prefix += 'AE'
        else:
            self.prefix += 'DAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if latent_act:
            self.prefix += 't'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix = self.prefix + 's'

        # filename for temporary model checkpoint
        modelName = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(modelName):
            os.remove(modelName)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X = self.X_train if self.pretrain_X is None else self.pretrain_X
        Y = self.y_train if self.pretrain_Y is None else self.pretrain_Y
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(X, Y, test_size=val_rate, random_state=self.seed, stratify=Y)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create autoencoder model
        self.autoencoder, self.encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)
        self.autoencoder.summary()

        if no_trn:
            return

        # compile model
        opt = Adam(learning_rate=lr)
        self.autoencoder.compile(optimizer=opt, loss=loss)

        # fit model
        self.history = self.autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                             verbose=verbose, validation_data=(X_inner_test, X_inner_test))
        # save loss progress
        self.saveLossProgress()

        # load best model
        self.autoencoder = load_model(modelName)
        layer_idx = int((len(self.autoencoder.layers) - 1) / 2)
        self.encoder = Model(self.autoencoder.layers[0].input, self.autoencoder.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)
        self.pretrain_X = self.encoder.predict(self.pretrain_X)

        # applying the learned encoder into the additional datasets:
        for i in range(len(self.additional_filenames)):
            self.additional_X[i] = self.encoder.predict(self.additional_X[i])
        self.printDataShapes()

    # Variational Autoencoder
    def vae(self, dims = [10], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, beta=1.0, warmup=True, warmup_rate=0.01, val_rate=0.2, no_trn=False, lr=0.0005):
        print("Training variational autoencoder")
        # manipulating an experiment identifier in the output file
        if patience != 25:
            self.prefix += 'p' + str(patience) + '_'
        if warmup:
            self.prefix += 'w' + str(warmup_rate) + '_'
        self.prefix += 'VAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        if beta != 1:
            self.prefix += 'B' + str(beta)
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        modelName = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(modelName):
            os.remove(modelName)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]

        # warm-up callback
        warm_up_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: [warm_up(epoch)])  # , print(epoch), print(K.get_value(beta))])

        # warm-up implementation
        def warm_up(epoch):
            val = epoch * warmup_rate
            if val <= 1.0:
                K.set_value(beta, val)
        # add warm-up callback if requested
        if warmup:
            beta = K.variable(value=0.0)
            callbacks.append(warm_up_cb)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X = self.X_train if self.pretrain_X is None else self.pretrain_X
        Y = self.y_train if self.pretrain_Y is None else self.pretrain_Y
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(X, Y,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=Y)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create vae model
        self.vae, self.encoder, self.decoder = DNN_models.variational_AE(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta, lr=lr)
        self.vae.summary()

        if no_trn:
            return

        # fit
        self.history = self.vae.fit(X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, None))

        # save loss progress
        self.saveLossProgress()

        # load best model
        self.vae.load_weights(modelName)
        self.encoder = self.vae.layers[1]

        # applying the learned encoder into the whole training and the test set.
        _, _, self.X_train = self.encoder.predict(self.X_train)
        _, _, self.X_test = self.encoder.predict(self.X_test)
        _, _, self.pretrain_X = self.encoder.predict(self.pretrain_X)

        # Apply the learned encoder to the additional datasets
        for i in range(len(self.additional_filenames)):
            _, _, self.additional_X[i] = self.encoder.predict(self.additional_X[i])

    # Convolutional Autoencoder
    def cae(self, dims = [32], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, val_rate=0.2, rf_rate = 0.1, st_rate = 0.25, no_trn=False, lr=0.0005):
        print("Training convolutional autoencoder")
        # manipulating an experiment identifier in the output file
        self.prefix += 'CAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        modelName = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(modelName):
            os.remove(modelName)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(modelName, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]


        # fill out blank
        X = self.X_train if self.pretrain_X is None else self.pretrain_X
        onesideDim = int(math.sqrt(X.shape[1])) + 1
        enlargedDim = onesideDim ** 2
        self.X_train = np.column_stack((self.X_train, np.zeros((self.X_train.shape[0], enlargedDim - self.X_train.shape[1]))))
        self.X_test = np.column_stack((self.X_test, np.zeros((self.X_test.shape[0], enlargedDim - self.X_test.shape[1]))))
        self.pretrain_X = np.column_stack((self.pretrain_X, np.zeros((self.pretrain_X.shape[0], enlargedDim - self.pretrain_X.shape[1]))))

        # reshape
        self.X_train = np.reshape(self.X_train, (len(self.X_train), onesideDim, onesideDim, 1))
        self.X_test = np.reshape(self.X_test, (len(self.X_test), onesideDim, onesideDim, 1))
        self.pretrain_X = np.reshape(self.pretrain_X, (len(self.pretrain_X), onesideDim, onesideDim, 1))

        # Apply the same transformations to the additional datasets
        for i in range(len(self.additional_filenames)):
            # onesideDim = int(math.sqrt(self.additional_X[i].shape[1])) + 1
            # enlargedDim = onesideDim ** 2
            self.additional_X[i] = np.column_stack((self.additional_X[i], np.zeros((self.additional_X[i].shape[0], enlargedDim - self.additional_X[i].shape[1]))))
            self.additional_X[i] = np.reshape(self.additional_X[i], (len(self.additional_X[i]), onesideDim, onesideDim, 1))

        self.printDataShapes()

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X = self.X_train if self.pretrain_X is None else self.pretrain_X
        Y = self.y_train if self.pretrain_Y is None else self.pretrain_Y
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(X, Y,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=Y)

        # insert input shape into dimension list
        dims.insert(0, (onesideDim, onesideDim, 1))

        # create cae model
        self.cae, self.encoder = DNN_models.conv_autoencoder(dims, act=act, output_act=output_act, rf_rate = rf_rate, st_rate = st_rate)
        self.cae.summary()
        if no_trn:
            return

        # compile
        opt = Adam(learning_rate=lr)
        self.cae.compile(optimizer=opt, loss=loss)

        # fit
        self.history = self.cae.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, X_inner_test, None))

        # save loss progress
        self.saveLossProgress()

        # load best model
        self.cae.load_weights(modelName)
        if len(self.cae.layers) % 2 == 0:
            layer_idx = int((len(self.cae.layers) - 2) / 2)
        else:
            layer_idx = int((len(self.cae.layers) - 1) / 2)
        self.encoder = Model(self.cae.layers[0].input, self.cae.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)
        self.pretrain_X = self.encoder.predict(self.pretrain_X)

        # applying the learned encoder into the additional datasets:
        for i in range(len(self.additional_filenames)):
            self.additional_X[i] = self.encoder.predict(self.additional_X[i])
        self.printDataShapes()

    # Classification
    def classification(self, hyper_parameters, method='svm', cv=5, scoring='roc_auc', n_jobs=1, cache_size=10000, check_val_test_correlation=True):
        """
        Perform classification on the dataset using the specified method and hyperparameters.

        This function conducts hyperparameter tuning using cross-validation, trains a classifier
        with the best parameters, and evaluates its performance on the test set. It also checks
        the correlation between validation and test set performance if specified.

        Parameters:
        -----------
        hyper_parameters : dict
            Dictionary of hyperparameters to search over during grid search.
        method : str, optional (default='svm')
            Classification method to use. Options: 'svm', 'rf', 'mlp'.
        cv : int, optional (default=5)
            Number of folds for cross-validation.
        scoring : str, optional (default='roc_auc')
            Scoring metric for hyperparameter tuning.
        n_jobs : int, optional (default=1)
            Number of parallel jobs to run for hyperparameter search.
        cache_size : int, optional (default=10000)
            Cache size for SVM (in MB).
        check_val_test_correlation : bool, optional (default=True)
            Whether to check correlation between validation and test set performance.

        Returns:
        --------
        None
            Results are printed and saved to files.

        Side Effects:
        -------------
        - Prints classification results and best parameters.
        - Saves performance metrics to a file.
        - If check_val_test_correlation is True, generates and saves a correlation plot.
        - Evaluates the best model on additional datasets if available.

        Notes:
        ------
        This function assumes that the data has already been loaded and preprocessed.
        It filters out unlabeled data before classification.
        """
        if check_val_test_correlation:
            # We check whether performance of the classifier on a validation set is correlated with its performance on a test set.
            # We first split X_train and Y_train into a training and validation set. Then, we train the classifier on the training set
            # and evaluate its performance on the validation and test sets and record the results. We repeat the process for different
            # val / train splits and classifier hyperparameters and then check the correlation between the validation and test set performance.
            # Assume random forest classifier for now.

            # Number of iterations for different splits and hyperparameters
            n_iterations = 50
            val_scores = []
            test_scores = []

            for iteration_n in range(n_iterations):
                # Split into train and validation sets
                valid_indices = np.where(self.y_train != -100)[0]
                X_train_labeled = self.X_train[valid_indices]
                y_train_labeled = self.y_train[valid_indices]

                # Create a separate validation set from the training data
                X_train_inner, X_val, y_train_inner, y_val = train_test_split(
                    X_train_labeled, y_train_labeled, test_size=0.2, stratify=y_train_labeled, random_state=self.seed + iteration_n
                )

                # Random hyperparameters for Random Forest
                n_estimators = np.random.randint(50, 200)
                max_depth = np.random.randint(3, 15)
                
                # Train Random Forest classifier
                rf = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=self.seed + iteration_n
                )
                rf.fit(X_train_inner, y_train_inner)

                # Evaluate on validation set
                val_prob = rf.predict_proba(X_val)[:, 1]
                val_score = roc_auc_score(y_val, val_prob)
                val_scores.append(val_score)

                # Evaluate on test set
                test_prob = rf.predict_proba(self.X_test)[:, 1]
                test_score = roc_auc_score(self.y_test, test_prob)
                test_scores.append(test_score)

            # Calculate correlation between validation and test scores
            correlation, p_value = pearsonr(val_scores, test_scores)

            print(f"Evaluating dataset {self.data} with prefux {self.prefix} and method {method}")
            print(f"Correlation between validation and test scores: {correlation:.4f}")
            print(f"P-value: {p_value:.4f}")

            if p_value < 0.05 and correlation > 0.5:
                print("There is a significant positive correlation between validation and test performance.")
                print("Proceeding with hyperparameter tuning using cross-validation.")
            else:
                print("Warning: Weak or no significant correlation between validation and test performance.")
                print("Cross-validation results may not reliably predict test set performance.")
                print("Consider collecting more data or trying different feature representations.")
            
            # Generate scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(val_scores, test_scores, alpha=0.6)
            plt.xlabel('Validation Scores')
            plt.ylabel('Test Scores')
            plt.title(f'Validation vs Test Scores\nCorrelation: {correlation:.4f}, p-value: {p_value:.4f}')
            
            # Add correlation line
            z = np.polyfit(val_scores, test_scores, 1)
            p = np.poly1d(z)
            plt.plot(val_scores, p(val_scores), "r--", alpha=0.8)
            
            # Add text box with correlation and p-value
            plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}\np-value: {p_value:.4f}', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Save plot as PDF
            plt.tight_layout()
            plt.savefig(f'results/plots/{self.data}_{self.prefix}_{method}_correlation_plot.pdf')
            plt.close()

            print(f"Correlation plot saved as {self.data}_{self.prefix}_{method}_correlation_plot.pdf")



        print("Starting classification")
        clf_start_time = time.time()

        print("# Tuning hyper-parameters")
        print(self.X_train.shape, self.y_train.shape)
        # Filter out unlabeled data elements from X_train / y_train
        valid_indices = np.where(self.y_train != -100)[0]
        self.X_train = self.X_train[valid_indices]
        self.y_train = self.y_train[valid_indices]
        print("Filtered data:", self.X_train.shape, self.y_train.shape)

        # Create a custom CV splitter that respects host boundaries
        if self.train_host_to_indices_dict is not None:
            cv_splitter = HostBlockedKFold(n_splits=cv, host_to_indices_dict=self.train_host_to_indices_dict, shuffle=True, random_state=self.seed)
        else:
            cv_splitter = StratifiedKFold(cv, shuffle=True, random_state=self.seed)


        # Support Vector Machine
        if method == 'svm':
            clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs, verbose=100, )
            clf.fit(self.X_train, self.y_train)

        # Random Forest
        if method == 'rf':
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), hyper_parameters, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train)

        # Multi-layer Perceptron
        if method == 'mlp':
            class_weights = {0: self.y_train.shape[0] / (self.y_train == 0).sum(), 1: self.y_train.shape[0] / (self.y_train == 1).sum()}
            model = KerasClassifier(build_fn=DNN_models.mlp_model, input_dim=self.X_train.shape[1], verbose=0, class_weight = class_weights)
            clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=cv_splitter, scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train, batch_size=32)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

        # Evaluate performance of the best model on test set
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        y_prob = clf.predict_proba(self.X_test)

        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score, AUPR, AUPR_neg
        metrics = [round(roc_auc_score(y_true, y_prob[:, 1]), 4),
                   round(accuracy_score(y_true, y_pred), 4),
                   round(recall_score(y_true, y_pred), 4),
                   round(precision_score(y_true, y_pred), 4),
                   round(f1_score(y_true, y_pred), 4),
                   round(average_precision_score(y_true, y_prob[:, 1]), 4),
                   round(average_precision_score(1 - y_true, y_prob[:, 0]), 4),
                   ]

        # time stamp
        metrics.append(str(datetime.datetime.now()))

        # running time
        metrics.append(round( (time.time() - self.t_start), 2))

        # classification time
        metrics.append(round( (time.time() - clf_start_time), 2))

        # best hyper-parameter append
        metrics.append(str(clf.best_params_))

        # Write performance metrics as a file
        res = pd.DataFrame([metrics], index=[self.prefix + method])
        pretrained_str = "" if self.pretrain_X is None else "_pretrained"
        test_inv_str = "_test_inv" if self.use_all_data_for_train else ""
        with open(self.data_dir + "results/" + self.data + pretrained_str + test_inv_str + "_result.txt", 'a') as f:
            res.to_csv(f, header=None)

        print('Accuracy metrics')
        print('AUC, ACC, Recall, Precision, F1_score, AUPR, AUPR_neg, time-end, runtime(sec), classfication time(sec), best hyper-parameter')
        print(metrics)

        # Evaluate the best model on each of the additional datasets
        for i in range(len(self.additional_filenames)):
            print(f"Evaluating on additional dataset: {self.additional_filenames[i]}")
            additional_y_true = self.additional_y[i]
            additional_y_pred = clf.predict(self.additional_X[i])
            additional_y_prob = clf.predict_proba(self.additional_X[i])

            additional_metrics = [round(roc_auc_score(additional_y_true, additional_y_prob[:, 1]), 4),
                                  round(accuracy_score(additional_y_true, additional_y_pred), 4),
                                  round(recall_score(additional_y_true, additional_y_pred), 4),
                                  round(precision_score(additional_y_true, additional_y_pred), 4),
                                  round(f1_score(additional_y_true, additional_y_pred), 4),
                                  round(average_precision_score(additional_y_true, additional_y_prob[:, 1]), 4),
                                  round(average_precision_score(1 - additional_y_true, additional_y_prob[:, 0]), 4),
                                  ]

            print(f"Metrics for {self.additional_filenames[i]}: AUC, ACC, Recall, Precision, F1_score, AUPR, AUPR_neg")
            print(additional_metrics)

            # Write the additional metrics to file
            res = pd.DataFrame([additional_metrics], index=[self.prefix + method + self.additional_filenames[i]])
            with open(self.data_dir + "results/" + self.data + pretrained_str + test_inv_str + "_result.txt", 'a') as f:
                res.to_csv(f, header=None)

    def printDataShapes(self, train_only=False):
        """
        Print the shapes of various data attributes of the DeepMicrobiome object.

        This function prints the shapes of training and test data, pretraining data if available,
        and information about host-to-indices dictionaries. It also prints shapes of additional datasets.

        Args:
            train_only (bool, optional): If True, only print information about training data. Defaults to False.

        Returns:
            None
        """
        print("X_train.shape: ", self.X_train.shape)
        if not train_only:
            print("y_train.shape: ", self.y_train.shape)
            print("X_test.shape: ", self.X_test.shape)
            print("y_test.shape: ", self.y_test.shape)
        if self.pretrain_X is not None:
            print("pretrain_X.shape: ", self.pretrain_X.shape)
        if self.host_to_indices_dict is not None:
            print("host_to_indices_dict length: ", len(self.host_to_indices_dict))
            print("num hosts: ", len(self.host_to_indices_dict.keys()))
            print("num samples: ", sum([len(indices) for indices in self.host_to_indices_dict.values()]))
        if self.train_host_to_indices_dict is not None:
            print("train_host_to_indices_dict length: ", len(self.train_host_to_indices_dict))
            print("num hosts: ", len(self.train_host_to_indices_dict.keys()))
            print("num samples: ", sum([len(indices) for indices in self.train_host_to_indices_dict.values()]))

        for i in range(len(self.additional_filenames)):
            print("Additional data: ", self.additional_filenames[i])
            print("additional_X.shape: ", self.additional_X[i].shape)
            print("additional_y_.shape: ", self.additional_y[i].shape)
            print()
    
    def split_data_by_host(self, X, Y, return_indices=False, test_size=0.2, random_state=0, stratify=None, host_to_indices_dict=None, stratify_tol=0.1, max_attempts=30):
        """
        Split the data into training and test sets while respecting host boundaries and maintaining class balance.

        This function attempts to split the data such that samples from the same host are not split between
        training and test sets, while also trying to maintain a similar class balance in both sets.

        Args:
            X (numpy.ndarray): Feature matrix.
            Y (numpy.ndarray): Labels.
            return_indices (bool, optional): If True, return indices along with split data. Defaults to False.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Random state for reproducibility. Defaults to 0.
            stratify (numpy.ndarray, optional): Array used for stratified splitting. If None, Y is used. Defaults to None.
            host_to_indices_dict (dict): Dictionary mapping hosts to their sample indices.
            stratify_tol (float, optional): Tolerance for class balance deviation. Defaults to 0.1.
            max_attempts (int, optional): Maximum number of splitting attempts. Defaults to 30.

        Returns:
            tuple: Depending on return_indices, returns either:
                (X_train, X_test, y_train, y_test, train_host_to_indices_dict, test_host_to_indices_dict) or
                (train_indices, test_indices, X_train, X_test, y_train, y_test, train_host_to_indices_dict, test_host_to_indices_dict)

        Raises:
            ValueError: If splitting fails to achieve desired class balance within max_attempts.
        """
        if host_to_indices_dict is None:
            raise ValueError("host_to_indices_dict is required")
        if stratify is None:
            # Assume stratify is Y if not provided
            print("Stratify is not provided, using Y as stratify")
            stratify = Y
        # Get unique hosts
        hosts = list(host_to_indices_dict.keys())

        # Find the host with the largest number of data points
        largest_host = max(host_to_indices_dict, key=lambda x: len(host_to_indices_dict[x]))
        largest_host_indices = host_to_indices_dict[largest_host]
        
        # Get the labels for the largest host
        largest_host_labels = Y[largest_host_indices]
        
        print(f"Host with the largest number of data points: {largest_host}")
        print(f"Number of data points: {len(largest_host_indices)}")
        print("Labels for this host:")
        print(largest_host_labels)

        # Assume binary classification
        target_class_balance = np.mean(Y[Y != -100])
        print("Target class balance: ", target_class_balance)
        # Keep attempting to split the data until the class balance is within the tolerance
        attempts = 0

        while attempts < max_attempts:
            # Hash the random state to generate a unique seed but reproducible seed for each attempt
            random_state_hash = hash((random_state, attempts)) % (2**32)
            # Split hosts into train and test sets
            train_hosts, test_hosts = train_test_split(hosts, test_size=test_size, random_state=random_state_hash)
            # Get indices for train and test sets
            train_indices = [idx for host in train_hosts for idx in host_to_indices_dict[host]]
            test_indices = [idx for host in test_hosts for idx in host_to_indices_dict[host]]
        
            # Split the data
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = Y[train_indices]
            y_test = Y[test_indices]

            # Create new host_to_indices dictionaries for train and test sets
            train_host_to_indices_dict = {}
            test_host_to_indices_dict = {}
            train_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(train_indices)}
            test_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(test_indices)}

            for host in train_hosts:
                train_host_to_indices_dict[host] = [train_index_map[idx] for idx in host_to_indices_dict[host] if idx in train_indices]

            for host in test_hosts:
                test_host_to_indices_dict[host] = [test_index_map[idx] for idx in host_to_indices_dict[host] if idx in test_indices]

            # Check how close the class balance is to the target
            current_train_class_balance = np.mean(y_train[y_train != -100])
            current_test_class_balance = np.mean(y_test[y_test != -100])
            print("Attempt {}: Train class balance: {:.4f}, Test class balance: {:.4f}".format(attempts, current_train_class_balance, current_test_class_balance))
            if (current_train_class_balance * (1 - stratify_tol) < target_class_balance and current_train_class_balance * (1 + stratify_tol) > target_class_balance) and \
               (current_test_class_balance * (1 - stratify_tol) < target_class_balance and current_test_class_balance * (1 + stratify_tol) > target_class_balance):
                if return_indices:
                    return train_indices, test_indices, X_train, X_test, y_train, y_test, train_host_to_indices_dict, test_host_to_indices_dict
                else:
                    return X_train, X_test, y_train, y_test, train_host_to_indices_dict, test_host_to_indices_dict
            attempts += 1
        raise ValueError("Failed to split data within the specified tolerance after {} attempts.".format(max_attempts))

    # ploting loss progress over epochs
    def saveLossProgress(self):
        """
        Save plots of the training and validation loss progress.

        This function creates and saves two plots:
        1. A plot of training and validation loss over epochs.
        2. If available, a detailed plot including reconstruction loss and KL divergence for VAEs.

        The plots are saved as PNG files in the results directory.

        Returns:
            None
        """
        #print(self.history.history.keys())
        #print(type(self.history.history['loss']))
        #print(min(self.history.history['loss']))

        loss_collector, loss_max_atTheEnd = self.saveLossProgress_ylim()

        # save loss progress - train and val loss only
        figureName = self.prefix + self.data + '_' + str(self.seed)
        plt.ylim(min(loss_collector)*0.9, loss_max_atTheEnd * 2.0)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'],
                   loc='upper right')
        plt.savefig(self.data_dir + "results/" + figureName + '.png')
        plt.close()

        if 'recon_loss' in self.history.history:
            figureName = self.prefix + self.data + '_' + str(self.seed) + '_detailed'
            plt.ylim(min(loss_collector) * 0.9, loss_max_atTheEnd * 2.0)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.plot(self.history.history['recon_loss'])
            plt.plot(self.history.history['val_recon_loss'])
            plt.plot(self.history.history['kl_loss'])
            plt.plot(self.history.history['val_kl_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train loss', 'val loss', 'recon_loss', 'val recon_loss', 'kl_loss', 'val kl_loss'], loc='upper right')
            plt.savefig(self.data_dir + "results/" + figureName + '.png')
            plt.close()

    # supporting loss plot
    def saveLossProgress_ylim(self):
        """
        Calculate the y-axis limits for the loss progress plots.

        This helper function determines appropriate y-axis limits for the loss plots
        by collecting all loss values and finding the maximum loss at the end of training.

        Returns:
            tuple: A tuple containing:
                - loss_collector (list): A list of all loss values.
                - loss_max_atTheEnd (float): The maximum loss value at the end of training.
        """
        loss_collector = []
        loss_max_atTheEnd = 0.0
        for hist in self.history.history:
            current = self.history.history[hist]
            loss_collector += current
            if current[-1] >= loss_max_atTheEnd:
                loss_max_atTheEnd = current[-1]
        return loss_collector, loss_max_atTheEnd

if __name__ == '__main__':
    """
    Main execution point for the DeepMicrobiome analysis pipeline.

    This script provides a command-line interface for running microbiome data analysis,
    including data loading, representation learning, and classification tasks.

    The script performs the following main steps:
    1. Parse command-line arguments to configure the analysis.
    2. Set up disease labels and hyperparameter grids for classifiers.
    3. Define and execute the experimental run function.
    4. Handle multiple experimental runs with different random seeds if specified.

    Usage:
        python DM.py [options]

    For a full list of available options, use:
        python DM.py -h

    The script is designed to be flexible, allowing users to specify various
    data sources, representation learning methods, and classification algorithms.
    It can handle both provided datasets and custom user data.

    Examples of how to invoke this script for the DeepMicro-based baseline experiments in "Learning a deep language model for microbiomes: the power of large scale unlabeled microbiome data" are show in the exps_cae.sh, exps_ae.sh, and exps_baselines.sh files.

    Raises:
        OSError: Logs any OS-related exceptions that occur during execution.

    See the argparse setup in the script for detailed information on available options.
    """
    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser._action_groups.pop()

    # load data
    load_data = parser.add_argument_group('Loading data')
    load_data.add_argument("-d", "--data", help="prefix of dataset to open (e.g. abundance_Cirrhosis)", type=str,
                        choices=["abundance_Cirrhosis", "abundance_Colorectal", "abundance_IBD",
                                 "abundance_Obesity", "abundance_T2D", "abundance_WT2D",
                                 "marker_Cirrhosis", "marker_Colorectal", "marker_IBD",
                                 "marker_Obesity", "marker_T2D", "marker_WT2D",
                                 ])
    load_data.add_argument("-cd", "--custom_data", help="filename for custom input data under the 'data' folder", type=str,)
    load_data.add_argument("-cl", "--custom_data_labels", help="filename for custom input labels under the 'data' folder", type=str,)
    load_data.add_argument("-ad", "--additional_data", help="comma seperated filenames for additional input data under the 'data' folder", type=str, default="")
    load_data.add_argument("-al", "--additional_data_labels", help="comma seperated filenames for additional input labels under the 'data' folder", type=str, default="")
    load_data.add_argument("-pd", "--pretraining_data", help="filenames for custom pretraining data under the 'data' folder, which should be structured like normal data, but with -100 as the label of any data point that doesn't have its own real label", type=str, default="")
    load_data.add_argument("-p", "--data_dir", help="custom path for both '/data' and '/results' folders", default="")
    load_data.add_argument("-dt", "--dataType", help="Specify data type for numerical values (float16, float32, float64)",
                        default="float64", type=str, choices=["float16", "float32", "float64"])
    load_data.add_argument("-htid", "--host_to_indices_dict_str", help="Specify host to indices dictionary",
                        default=None, type=str)
    
    dtypeDict = {"float16": np.float16, "float32": np.float32, "float64": np.float64}

    # experiment design
    exp_design = parser.add_argument_group('Experiment design')
    exp_design.add_argument("-s", "--seed", help="random seed for train and test split", type=int, default=0)
    exp_design.add_argument("-r", "--repeat", help="repeat experiment x times by changing random seed for splitting data",
                        default=5, type=int)

    # classification
    classification = parser.add_argument_group('Classification')
    classification.add_argument("-f", "--numFolds", help="The number of folds for cross-validation in the tranining set",
                        default=5, type=int)
    classification.add_argument("-m", "--method", help="classifier(s) to use", type=str, default="all",
                        choices=["all", "svm", "rf", "mlp", "svm_rf"])
    classification.add_argument("-sc", "--svm_cache", help="cache size for svm run", type=int, default=1000)
    classification.add_argument("-t", "--numJobs",
                        help="The number of jobs used in parallel GridSearch. (-1: utilize all possible cores; -2: utilize all possible cores except one.)",
                        default=-2, type=int)
    parser.add_argument("--scoring", help="Metrics used to optimize method", type=str, default='roc_auc',
                        choices=['roc_auc', 'accuracy', 'f1', 'recall', 'precision', 'average_precision'])

    # representation learning & dimensionality reduction algorithms
    rl = parser.add_argument_group('Representation learning')
    rl.add_argument("--pca", help="run PCA", action='store_true')
    rl.add_argument("--rp", help="run Random Projection", action='store_true')
    rl.add_argument("--ae", help="run Autoencoder or Deep Autoencoder", action='store_true')
    rl.add_argument("--vae", help="run Variational Autoencoder", action='store_true')
    rl.add_argument("--cae", help="run Convolutional Autoencoder", action='store_true')
    rl.add_argument("--save_rep", help="write the learned representation of the training set as a file", action='store_true')

    # detailed options for representation learning
    ## common options
    common = parser.add_argument_group('Common options for representation learning (SAE,DAE,VAE,CAE)')
    common.add_argument("--aeloss", help="set autoencoder reconstruction loss function", type=str,
                        choices=['mse', 'binary_crossentropy'], default='mse')
    common.add_argument("--ae_oact", help="output layer sigmoid activation function on/off", action='store_true')
    common.add_argument("-a", "--act", help="activation function for hidden layers", type=str, default='relu',
                        choices=['relu', 'sigmoid'])
    common.add_argument("-dm", "--dims",
                        help="Comma-separated dimensions for deep representation learning e.g. (-dm 50,30,20)",
                        type=str, default='50')
    common.add_argument("-e", "--max_epochs", help="Maximum epochs when training autoencoder", type=int, default=2000)
    common.add_argument("-pt", "--patience",
                        help="The number of epochs which can be executed without the improvement in validation loss, right after the last improvement.",
                        type=int, default=20)
    common.add_argument("-lr", "--learning_rate", help="learning rate for optimizer", type=float, default=0.0005)

    ## AE & DAE only
    AE = parser.add_argument_group('SAE & DAE-specific arguments')
    AE.add_argument("--ae_lact", help="latent layer activation function on/off", action='store_true')

    ## VAE only
    VAE = parser.add_argument_group('VAE-specific arguments')
    VAE.add_argument("--vae_beta", help="weight of KL term", type=float, default=1.0)
    VAE.add_argument("--vae_warmup", help="turn on warm up", action='store_true')
    VAE.add_argument("--vae_warmup_rate", help="warm-up rate which will be multiplied by current epoch to calculate current beta", default=0.01, type=float)

    ## CAE only
    CAE = parser.add_argument_group('CAE-specific arguments')
    CAE.add_argument("--rf_rate", help="What percentage of input size will be the receptive field (kernel) size? [0,1]", type=float, default=0.1)
    CAE.add_argument("--st_rate", help="What percentage of receptive field (kernel) size will be the stride size? [0,1]", type=float, default=0.25)

    # other options
    others = parser.add_argument_group('other optional arguments')
    others.add_argument("--no_trn", help="stop before learning representation to see specified autoencoder structure", action='store_true')
    others.add_argument("--no_clf", help="skip classification tasks", action='store_true')
    others.add_argument("--use_all_data_for_train", help="use all data for training, should only be used when an additional dataset for testing is provided for testing", action='store_true')


    args = parser.parse_args()
    print(args)

    # set labels for diseases and controls
    label_dict = {
        # Controls
        'n': 0,
        # Chirrhosis
        'cirrhosis': 1,
        # Colorectal Cancer
        'cancer': 1, 'small_adenoma': 0,
        # IBD
        'ibd_ulcerative_colitis': 1, 'ibd_crohn_disease': 1,
        # T2D and WT2D
        't2d': 1,
        # Obesity
        'leaness': 0, 'obesity': 1,
    }

    # hyper-parameter grids for classifiers
    rf_hyper_parameters = [{'n_estimators': [s for s in range(100, 1001, 200)],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_leaf': [1, 2, 3, 4, 5],
                            'criterion': ['gini', 'entropy']
                            }, ]
    #svm_hyper_parameters_pasolli = [{'C': [2 ** s for s in range(-5, 16, 2)], 'kernel': ['linear']},
    #                        {'C': [2 ** s for s in range(-5, 16, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
    #                         'kernel': ['rbf']}]
    svm_hyper_parameters = [{'C': [2 ** s for s in range(-5, 6, 2)], 'kernel': ['linear']},
                            {'C': [2 ** s for s in range(-5, 6, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],'kernel': ['rbf']}]
    mlp_hyper_parameters = [{'numHiddenLayers': [1, 2, 3],
                             'epochs': [30, 50, 100, 200, 300],
                             'numUnits': [10, 30, 50, 100],
                             'dropout_rate': [0.1, 0.3],
                             },]


    # run exp function
    def run_exp(seed):
        """
        Run a single experiment with the specified seed and configuration.

        This function sets up and executes a complete experiment pipeline, including:
        - Loading data (either provided or custom)
        - Performing representation learning (if specified)
        - Running classification tasks (if not disabled)

        The function uses global arguments from the argparse setup to configure
        the experiment parameters.

        Parameters:
        -----------
        seed : int
            Random seed for reproducibility.

        Global Arguments:
        -----------------
        args : argparse.Namespace
            Command-line arguments parsed by argparse, including:
            - data: Name of the provided dataset
            - custom_data: Name of the custom input data file
            - custom_data_labels: Name of the custom input labels file
            - additional_data: Names of additional input data files
            - additional_data_labels: Names of additional input label files
            - pretraining_data: Name of pretraining data file
            - pca, ae, rp, vae, cae: Flags for different representation learning methods
            - no_clf: Flag to skip classification
            - method: Classification method to use
            ... (and many other configuration parameters)

        Returns:
        --------
        None

        Side Effects:
        -------------
        - Creates and manipulates a DeepMicrobiome object
        - Loads data into memory
        - Performs representation learning (if specified)
        - Runs classification tasks (if not disabled)
        - Writes results to files

        Raises:
        -------
        ValueError
            If invalid combinations of arguments are provided.

        Notes:
        ------
        This function is designed to be called multiple times with different seeds
        for repeated experiments. It relies heavily on the global configuration
        set up through command-line arguments.
        """
        print(args.host_to_indices_dict_str)

        # create an object and load data
        ## no argument founded
        if args.data == None and args.custom_data == None:
            print("[Error] Please specify an input file. (use -h option for help)")
            exit()
        ## provided data
        elif args.data != None:
            if args.use_all_data_for_train:
                raise ValueError("use_all_data_for_train is not supported for standard training; must provide additional data to act as test set")
            dm = DeepMicrobiome(data=args.data + '.txt', seed=seed, data_dir=args.data_dir, host_to_indices_dict_str=args.host_to_indices_dict_str)

            ## specify feature string
            feature_string = ''
            data_string = str(args.data)
            if data_string.split('_')[0] == 'abundance':
                feature_string = "k__"
            if data_string.split('_')[0] == 'marker':
                feature_string = "gi|"

            ## load data into the object
            dm.loadData(feature_string=feature_string, label_string='disease', label_dict=label_dict,
                        dtype=dtypeDict[args.dataType])

        ## user data
        elif args.custom_data != None:
            print("Starting to load custom data")

            ### without labels - only conducting representation learning
            if args.custom_data_labels == None:
                dm = DeepMicrobiome(data=args.custom_data, seed=seed, data_dir=args.data_dir, host_to_indices_dict_str=args.host_to_indices_dict_str)
                dm.loadCustomData(dtype=dtypeDict[args.dataType])

            ### with labels - conducting representation learning + classification
            else:
                dm = DeepMicrobiome(data=args.custom_data, seed=seed, data_dir=args.data_dir, host_to_indices_dict_str=args.host_to_indices_dict_str)
                dm.loadCustomDataWithLabels(label_data=args.custom_data_labels, dtype=dtypeDict[args.dataType], use_all_data_for_train=args.use_all_data_for_train)
                #### with additional datasets for testing
                if args.additional_data != "":
                    additional_data = args.additional_data.split(',')
                    additional_data_labels = args.additional_data_labels.split(',')
                    dm.loadAdditionalData(additional_data, additional_data_labels, dtype=dtypeDict[args.dataType])
        if args.pretraining_data != "":
            dm.loadPretrainingData(args.pretraining_data, dtype=dtypeDict[args.dataType])

        numRLrequired = min(1, args.pca + args.ae + args.rp + args.vae + args.cae)

        #if numRLrequired > 1:
        #    raise ValueError('No multiple dimensionality Reduction')

        # time check after data has been loaded
        dm.t_start = time.time()

        # Representation learning (Dimensionality reduction)
        if args.pca:
            dm.pca()
        if args.ae:
            dm.ae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss,
                  latent_act=args.ae_lact, output_act=args.ae_oact, patience=args.patience, no_trn=args.no_trn, lr=args.learning_rate)
        if args.vae:
            dm.vae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                   patience= 25 if args.patience==20 else args.patience, beta=args.vae_beta, warmup=args.vae_warmup, warmup_rate=args.vae_warmup_rate, no_trn=args.no_trn, lr=args.learning_rate)
        if args.cae:
            dm.cae(dims=[int(i) for i in args.dims.split(',')], act=args.act, epochs=args.max_epochs, loss=args.aeloss, output_act=args.ae_oact,
                   patience=args.patience, rf_rate = args.rf_rate, st_rate = args.st_rate, no_trn=args.no_trn, lr=args.learning_rate)
        if args.rp:
            dm.rp()

        # write the learned representation of the training set as a file
        if args.save_rep:
            if numRLrequired == 1:
                rep_file = dm.data_dir + "results/" + dm.prefix + dm.data + "_rep.csv"
                pd.DataFrame(dm.X_train).to_csv(rep_file, header=None, index=None)
                print("The learned representation of the training set has been saved in '{}'".format(rep_file))
            else:
                print("Warning: Command option '--save_rep' is not applied as no representation learning or dimensionality reduction has been conducted.")

        # Classification
        if args.no_clf or (args.data == None and args.custom_data_labels == None):
            print("Classification task has been skipped.")
        else:
            # turn off GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            importlib.reload(keras)

            # training classification models
            if args.method == "svm":
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
            elif args.method == "rf":
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            elif args.method == "mlp":
                dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            elif args.method == "svm_rf":
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
            else:
                dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring, cache_size=args.svm_cache)
                dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)
                dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=args.numFolds,
                                  n_jobs=args.numJobs, scoring=args.scoring)



    # run experiments
    try:
        if args.repeat > 1:
            for i in range(args.repeat):
                run_exp(i + args.seed)
        else:
            run_exp(args.seed)

    except OSError as error:
        exception_handle.log_exception(error)
