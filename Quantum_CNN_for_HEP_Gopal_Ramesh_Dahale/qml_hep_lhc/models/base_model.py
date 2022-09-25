from tensorflow.keras import Model, losses, optimizers
from tensorflow.keras.metrics import AUC
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead
from qml_hep_lhc.models.metrics import custom_accuracy, qAUC


class BaseModel(Model):
    def __init__(self, args=None):
        super().__init__()
        self.args = vars(args) if args is not None else {}

        # Loss function
        self.loss = self.args.get('loss', "CategoricalCrossentropy")
        self.loss_fn = getattr(losses, self.loss)()

        self.lr = self.args.get('learning_rate', 0.002)

        # Optimizer
        if self.args.get('optimizer', 'Adam') == 'Adam':
            self.optimizer = getattr(optimizers, 'Adam')(learning_rate=self.lr)
        elif self.args.get('optimizer', 'Adam') == 'Ranger':
            radam = RectifiedAdam(learning_rate=self.lr)
            self.optimizer = Lookahead(radam,
                                       sync_period=6,
                                       slow_step_size=0.5)
        else:
            self.optimizer = getattr(
                optimizers, self.args.get('optimizer'))(learning_rate=self.lr)

        # Learning rate scheduler
        self.batch_size = self.args.get('batch_size', 128)

        if self.args.get('use_quantum', False):
            self.loss = "MeanSquaredError"
            self.loss_fn = getattr(losses, self.loss)()
            self.accuracy = [qAUC(), custom_accuracy]
            self.acc_metrics = ['custom_accuracy,', 'qAUC']
        else:
            # Accuracy
            self.accuracy = [AUC(), 'accuracy']
            self.acc_metrics = ['accuracy,', 'AUC']

    def compile(self):
        super(BaseModel, self).compile(loss=self.loss_fn,
                                       metrics=self.accuracy,
                                       optimizer=self.optimizer,
                                       run_eagerly=True)

    def fit(self, data, callbacks):
        return super(BaseModel,
                     self).fit(data.train_ds,
                               batch_size=self.batch_size,
                               epochs=self.args.get('epochs', 3),
                               callbacks=callbacks,
                               validation_data=data.val_ds,
                               shuffle=True,
                               workers=self.args.get('num_workers', 4))

    def test(self, data, callbacks):
        return super(BaseModel,
                     self).evaluate(data.test_ds,
                                    callbacks=callbacks,
                                    batch_size=self.batch_size,
                                    workers=self.args.get('num_workers', 4))

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", "-opt", type=str, default="Adam")
        parser.add_argument("--learning-rate", "-lr", type=float, default=0.01)
        parser.add_argument("--loss",
                            "-l",
                            type=str,
                            default="CategoricalCrossentropy")
        parser.add_argument("--use-quantum",
                            "-q",
                            action="store_true",
                            default=False)
        return parser
