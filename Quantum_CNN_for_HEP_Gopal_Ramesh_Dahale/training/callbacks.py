import wandb
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow import concat
from tensorflow import map_fn
from os import path, makedirs
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from qml_hep_lhc.data.utils import tf_ds_to_numpy


# ROC Plot callback for wandb
class PRMetrics(Callback):

    def __init__(self, data, use_quantum):
        self.x, self.y = tf_ds_to_numpy(data.test_ds)
        self.use_quantum = use_quantum
        self.classes = data.classes

    def on_train_end(self, logs=None):
        out = self.model.predict(self.x)
        if self.use_quantum:
            preds = map_fn(lambda x: 1.0 if x >= 0.5 else 0, out)
            probs = out
            probs = concat((1 - probs, probs), axis=1)
        else:
            self.y = self.y.argmax(axis=1)
            preds = out.argmax(axis=1)
            probs = out

        roc_curve = wandb.sklearn.plot_roc(self.y, probs, self.classes)
        confusion_matrix = wandb.sklearn.plot_confusion_matrix(
            self.y, preds, self.classes)

        wandb.log({"roc_curve": roc_curve})
        wandb.log({"confusion_matrix": confusion_matrix})


def _setup_callbacks(args, config, data):
    """
    This function initializes and returns a list of callbacks

    Args:
      args: This is the namespace object that contains all the parameters that we passed in from the
    command line.

    Returns:
      A list of callbacks.
    """

    callbacks = []

    # Wandb callback
    if args.wandb:
        if args.load_latest_checkpoint or (args.load_checkpoint is not None):
            if args.run_id is not None:
                run_id = args.run_id
        else:
            run_id = wandb.util.generate_id()

        print("RUN ID", run_id)
        wandb.init(project='qml-hep-lhc',
                   config=config,
                   id=run_id,
                   resume='allow')

        callbacks.append(
            wandb.keras.WandbCallback(save_weights_only=False,
                                      save_graph=False,
                                      save_model=False))
        callbacks.append(PRMetrics(data, args.use_quantum))

    try:
        ansatz = config['Base Model Args']['ansatz']
        feature_map = config['Base Model Args']['feature_map']
        checkpoint_path = f'{args.checkpoints_dir}/{args.data_class}/{args.model_class}/{ansatz}/{feature_map}'
    except:
        checkpoint_path = f'{args.checkpoints_dir}/{args.data_class}/{args.model_class}'

    checkpoint_dir = path.dirname(checkpoint_path)

    if args.save_checkpoint:
        if not path.exists(checkpoint_dir):
            makedirs(checkpoint_dir)

        filepath = f"{checkpoint_path}/cp.ckpt"

        # Create a callback that saves the model's weights
        model_checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                                    monitor='val_loss',
                                                    mode='min',
                                                    verbose=1,
                                                    save_weights_only=True,
                                                    save_freq='epoch')
        callbacks.append(model_checkpoint_callback)

    # LR Scheduler callback
    lr_scheduler_callback = ReduceLROnPlateau(monitor='val_loss',
                                              factor=np.sqrt(0.1),
                                              patience=5,
                                              min_delta=0.0001,
                                              min_lr=1e-8)

    # Early Stopping Callback
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            mode="min",
                                            patience=20)

    # Append callbacks
    callbacks.append(lr_scheduler_callback)
    # callbacks.append(early_stopping_callback)
    return callbacks, checkpoint_dir