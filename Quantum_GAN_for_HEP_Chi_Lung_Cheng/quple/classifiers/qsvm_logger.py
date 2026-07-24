import os
import time
import datetime
import logging

import numpy as np
import quple

from sklearn.metrics import accuracy_score,roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DefaultDict(dict):
    def __missing__(self, key):
        return key
    
class QSVMLogger():
    """Logger for the quantum support vector machine classifer (QSVM)
    
    """
    def __init__(self, 
                 log_dir='./logs', 
                 filename='qsvm_'
                 'circuit_{circuit_name}_'
                 'encoder_{encoding_map}_'
                 'entanglement_{entangle_strategy}_'
                 'nqubit_{n_qubit}_'
                 'train_{train_size}_'
                 'test_{test_size}_{time}',
                 keys=None,
                 stream_level=logging.INFO,
                 file_level=logging.DEBUG,
                 formatter=None,
                 save_npz=True,
                 save_weights=True,
                 roc_plot=True,
                 extra_attributes={}):
        """Creates the VQC logger
        Args:
            log_dir: str
                Name of logging directory.
            filename: str
                File name prefix for logged items.
            keys: str
                Keys of epoch results to save.
            stream_level: str
                Stream level for the logger.
            file_level: str
                File level for the logging file.
            formatter: str
                Formatter for logging.
            save_npz: boolean; default=True
                Whether to save VQC specs, epoch results and test results
                to a npz file.
            save_weights: boolean; default=True
                Whether to save the weights of the model.
            roc_plot: boolean; default=True
                Whether to save a roc plot for the test result.
            extra_attributes: dict, defaul={}
                Extra attributes to be saved in the logger
        """
        
        self.log_dir = log_dir
        self.filename = filename
        self.keys = keys
        self.stream_level = stream_level
        self.file_level = file_level
        self.stream_handler = None
        self.file_handler = None
        self.formatter = formatter or logging.Formatter('%(asctime)s [%(threadName)-12.12s]'
                                                         '[%(levelname)-5.5s]  %(message)s')
        self.logger_is_set = False
        self.save_npz = save_npz
        self.save_weights = save_weights
        self.roc_plot = roc_plot
        self.attributes = {}
        self.extra_attributes = extra_attributes
        
    def set_attributes(self, attributes):
        self.attributes = {**attributes, **self.extra_attributes}
        
    def reset_logger(self):
        self.file_handler.close()
        logger.removeHandler(self.stream_handler)
        logger.removeHandler(self.file_handler)
        self.logger_is_set = False
        
    def setup_logger(self):
        if self.logger_is_set:
            return None
        
        # setup time attributes
        self.start_time = time.time()
        self.time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S-%f')  
        
        # setup stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(self.formatter)
        self.stream_handler = stream_handler        
        
        # setup file name and formats
        self.attributes['time'] = self.time
        self.attributes = DefaultDict(self.attributes)
        self.formatted_filename = os.path.join(self.log_dir, self.filename.format(**self.attributes))
        log_filename = self.formatted_filename + '.log'
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(self.formatter)
        self.file_handler = file_handler
        
        #setup logger
        logger.addHandler(self.stream_handler)
        logger.addHandler(self.file_handler)
        self.logger_is_set = True
        
    def print_attributes(self):
        attrib = self.attributes.copy()
        logger.info('######## Executing QSVM with the following attributes ########')
        logger.info('Feature Dimension: {}'.format(attrib.pop('feature_dimension','')))            
        logger.info('Number of Qubits: {}'.format(attrib.pop('n_qubit','')))
        logger.info('Encoding Circuit: {}'.format(attrib.pop('encoding_circuit','')))
        logger.info('Encoding Map: {}'.format(attrib.pop('encoding_map','')))                
        logger.info('Train Size: {}'.format(attrib.pop('train_size','')))
        logger.info('Test Size: {}'.format(attrib.pop('test_size','')))
        for k,v in attrib.items():
            logger.info('{}: {}'.format(k.replace('_',' ').title(),v)) 
        logger.info('##############################################################')
        
    def info(self, text):
        logger.info(text)
        
    def on_train_start(self):
        self.setup_logger()
        self.print_attributes()
        
        
    def on_train_end(self, clf, test_kernel_matrix, y_test, val_kernel_matrix=None, y_val=None):
        logger.info('##################### Training Ends ##########################')
        
        score = clf.predict_proba(test_kernel_matrix)[:,1]
        logger.info("best_params: %s" % clf.best_params_)

        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        predictions = [round(value) for value in score]

        roc_auc =  roc_auc_score(y_test, score)
        accuracy =  accuracy_score(y_test, predictions)

        logger.info("AUC: %s" % roc_auc)
        logger.info("Accuracy: %s" % accuracy)
        
        self.end_time = time.time()
        
        self.attributes['time_taken'] = self.end_time - self.start_time
        
        result = {'score': score,
                  'y_test': y_test,
                  'auc': roc_auc,
                  'accuracy':accuracy}
        
        extras = {}
            
        if (val_kernel_matrix is not None) and (y_val is not None):
            val_score = clf.predict_proba(val_kernel_matrix)[:,1]
            val_auc =  roc_auc_score(y_val, val_score)
            result.update({'val_score': val_score,
                           'y_val': y_val,
                           'val_auc': val_auc}) 
                
        logger.info("Time taken: %s" % self.attributes['time_taken'])
        np.savez(self.formatted_filename +'.npz', 
                 result=result,
                 **self.attributes,
                 **extras)
        
        self.reset_logger()
        return roc_auc

    
    