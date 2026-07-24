import os
import time
import datetime
import logging

import numpy as np
import tensorflow as tf

import quple

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DefaultDict(dict):
    def __missing__(self, key):
        return key
    
class VQCLogger(tf.keras.callbacks.Callback):
    """Logger for the variational quantum classifier (VQC)
    
    """
    def __init__(self, log_dir='./logs', 
                 filename='{encoding_circuit}_{encoding_map}_'
                 '{variational_circuit}_{n_qubit}_qubit_'
                 '{activation}_{optimizer}_train_{train_size}_'
                 'val_{val_size}_test_{test_size}_epoch_{epochs}_'
                 'batch_size_{batch_size}_{time}',
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
            save_weights boolean; default=True
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
        self.time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
        self.logger_is_set = False
        self.train_end = False
        self.save_npz = save_npz
        self.save_weights = save_weights
        self.roc_plot = roc_plot
        self.attributes = {}
        self.extra_attributes = extra_attributes        
        super(VQCLogger, self).__init__() 
        self.set_params({'do_validation':False})

    def set_attributes(self, attributes):
        self.attributes = {**attributes, **self.extra_attributes}
        
    def setup_logger(self):
        if self.logger_is_set:
            return
        # setup stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(self.formatter)
        
        # setup file handler
        self.attributes['time'] = self.time
        self.attributes = DefaultDict(self.attributes)
        self.formatted_filename = os.path.join(self.log_dir, self.filename.format(**self.attributes))
        log_filename = self.formatted_filename + '.log'
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(self.formatter)
        self.stream_handler = stream_handler
        self.file_handler = file_handler
        
        #setup logger
        logger.addHandler(self.stream_handler)
        logger.addHandler(self.file_handler)
        self.logger_is_set = True
        self.print_attributes()
        
    def print_attributes(self):
        attrib = self.attributes.copy()
        logger.info('######## Executing VQC with the following attributes ########')
        logger.info('Feature Dimension: {}'.format(attrib['feature_dimension']))            
        logger.info('Number of Qubits: {}'.format(attrib['n_qubit']))
        logger.info('Qubit Layout: {}'.format(list(attrib['qubits'])))
        logger.info('Encoding Circuit: {}'.format(attrib['encoding_circuit']))
        logger.info('Encoding Map: {}'.format(attrib['encoding_map']))         
        logger.info('Variational Circuit: {}'.format(attrib['variational_circuit']))            
        logger.info('Circuit Parameters: {}'.format(list(attrib['circuit_parameters'])))
        logger.info('Number of Parameters: {}'.format(attrib['num_parameters']))            
        logger.info('Optimizer: {}'.format(attrib['optimizer']))
        logger.info('Metrics: {}'.format(list(attrib['metrics']))) 
        logger.info('Loss Function: {}'.format(attrib['loss'])) 
        logger.info('Activation Function: {}'.format(attrib['activation'])) 
        logger.info('Training Size: {}'.format(attrib['train_size']))
        logger.info('Validation Size: {}'.format(attrib['val_size']))
        logger.info('Test Size: {}'.format(attrib['test_size'])) 
        logger.info('Batch Size: {}'.format(attrib['batch_size'])) 
        logger.info('Number of Epochs: {}'.format(attrib['epochs'])) 
        
        logger.debug('Circuit Diagram for Encoding Circuit:\n{}'.format(self.model.encoding_circuit))
        logger.debug('Circuit Diagram for Variational Circuit:\n{}'.format(self.model.variational_circuit))
                    
    def reset_logger(self):
        self.train_end = False
        self.file_handler.close()
        logger.removeHandler(self.stream_handler)
        logger.removeHandler(self.file_handler)
        self.logger_is_set = False
    
    def on_train_begin(self, logs=None):
        self.setup_logger()  
        self.train_result = []
        logger.info('######## Training Begins ########')
        logger.info('Number of samples for Training: {}'.format(self.attributes['train_size']))
        logger.info('Number of Epochs: {}'.format(self.attributes['epochs']))
        logger.info('Batch Size: {}'.format(self.attributes['batch_size']))
        
    def on_train_end(self, logs=None):
        logger.info('######## Training Ends ########')
        weights = self.model.get_weights()[0]
        logger.info('Model weights: \n{}'.format(weights))
        if self.save_npz:
            filename = self.formatted_filename +'.npz'
            self.savez(filename, 
                     train_result=self.train_result,
                     model_weights=weights,
                     **self.model.attributes)
        
        if self.save_weights:
            weights_filename = self.formatted_filename + '_model_weights'
            self.model.save_weights(weights_filename)
        self.train_end = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}  
        
        # log all keys in logs if not specified
        if self.keys is None:
              self.keys = sorted(logs.keys())
        
        # set value to np.NaN if value is not available in last epoch
        if self.model.stop_training:
            logs = dict((k, logs[k]) if k in logs else (k, np.NaN) for k in self.keys)       
                    
        logger.debug('######## Epoch {} ########'.format(epoch))
        for key in logs:
            logger.debug('{}: {}'.format(key, logs[key]))
        logs['epoch'] = epoch    
        self.train_result.append(logs)
        
    def on_test_begin(self, logs=None):
        # only log when it's calling model.evaluate
        if not self.train_end:
            return 
        logger.info('######## Test Begins ########')
        logger.info('Number of samples for Testing: {}'.format(self.attributes['test_size']))
        logger.info('Number of Epochs: {}'.format(self.attributes['epochs']))
        logger.info('Batch Size: {}'.format(self.attributes['batch_size']))
    
    def savez(self, filename, **kwargs):
        temp = {}
        if os.path.exists(filename):
            temp = dict(np.load(filename, allow_pickle=True))
        temp.update(kwargs)
        np.savez(filename, **temp)
        
    def on_test_end(self, logs=None):
        logs = logs or {} 
        # only log when it's calling model.evaluate
        if not self.train_end:
            return 
        logger.info('######## Test Ends ########')
        
        if self.save_npz and logs:
            filename = self.formatted_filename +'.npz'
            self.savez(filename, test_result=logs)
 
                    

    def log_roc_curve(self, fpr, tpr, auc):
        logger.debug('######## ROC curve information ########')
        logger.debug('fpr:\n{}'.format(fpr))
        logger.debug('tpr:\n{}'.format(tpr))
        logger.debug('auc:\n{}'.format(auc))
        roc_result = {'fpr':fpr, 'tpr':tpr, 'auc':auc}
        if self.save_npz:
            filename = self.formatted_filename +'.npz'
            self.savez(filename, roc_result=roc_result)
        if self.roc_plot:
            import matplotlib
            matplotlib.use('Agg') 
            plt = quple.utils.utils.plot_roc_curve(fpr, tpr)
            filename = self.formatted_filename+'_roc_curve'
            plt.tight_layout()
            plt.savefig(filename+'.png')
            plt.savefig(filename+'.eps')
                    

    
    