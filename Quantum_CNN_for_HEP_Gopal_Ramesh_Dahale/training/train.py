from tensorflow import train
from argparse import ArgumentParser
from qml_hep_lhc.utils import _import_class
from callbacks import _setup_callbacks


def _setup_parser():
    """
    It creates a parser object, and then adds arguments to it

    Returns:
      A parser object
    """
    parser = ArgumentParser()

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--data-class", "-dc", type=str, default="MNIST")
    parser.add_argument("--model-class", "-mc", type=str, default="ResnetV1")
    parser.add_argument("--save-checkpoint",
                        "-sc",
                        action="store_true",
                        default=False)
    parser.add_argument("--checkpoints-dir",
                        "-cd",
                        type=str,
                        default="./checkpoints")
    parser.add_argument("--load-checkpoint", "-lc", type=str, default=None)
    parser.add_argument("--load-latest-checkpoint",
                        "-llc",
                        action="store_true",
                        default=False)

    temp_args, _ = parser.parse_known_args()

    base_data_class = _import_class(f"qml_hep_lhc.data.BaseDataModule")
    data_class = _import_class(f"qml_hep_lhc.data.{temp_args.data_class}")
    base_model_class = _import_class(f"qml_hep_lhc.models.BaseModel")
    model_class = _import_class(f"qml_hep_lhc.models.{temp_args.model_class}")
    dp_class = _import_class(f"qml_hep_lhc.data.DataPreprocessor")

    # Get data, model, and LitModel specific arguments
    base_data_group = parser.add_argument_group("Base Data Args")
    base_data_class.add_to_argparse(base_data_group)

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    base_model_group = parser.add_argument_group("Base Model Args")
    base_model_class.add_to_argparse(base_model_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    dp_group = parser.add_argument_group("Data Preprocessing Args")
    dp_class.add_to_argparse(dp_group)

    # model.fit specific arguments
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--num-workers", "-workers", type=int, default=4)
    return parser


def get_configuration(parser, args, data, model):
    arg_grps = {}
    for group in parser._action_groups:
        group_dict = {
            a.dest: getattr(args, a.dest, None) for a in group._group_actions
        }
        arg_grps[group.title] = group_dict

    # Add additional configurations
    arg_grps['Base Model Args']['loss'] = model.loss
    arg_grps['Base Model Args']['accuracy'] = model.acc_metrics
    arg_grps['Base Model Args']['scheduler'] = 'ReduceLROnPlateau'

    # Additional configurations for quantum model
    if hasattr(model, "fm_class"):
        arg_grps['Base Model Args']['feature_map'] = model.fm_class
    if hasattr(model, "ansatz_class"):
        arg_grps['Base Model Args']['ansatz'] = model.ansatz_class

    return arg_grps


def main():

    # Parsing the arguments from the command line.
    parser = _setup_parser()
    args = parser.parse_args()

    # Importing the data class
    data_class = _import_class(f"qml_hep_lhc.data.{args.data_class}")

    # Creating a data object, and then calling the prepare_data and setup methods on it.
    data = data_class(args)
    data.prepare_data()
    data.setup()

    print(data)

    # Importing the model class
    model_class = _import_class(f"qml_hep_lhc.models.{args.model_class}")
    model = model_class(data.config(), args)  # Model

    config = get_configuration(parser, args, data, model)
    callbacks, checkpoint_dir = _setup_callbacks(args, config, data)

    if args.load_latest_checkpoint:
        latest = train.latest_checkpoint(checkpoint_dir)
        print("Loading latest checkpoint from", latest)
        model.load_weights(latest)
        print("Loaded latest checkpoint")

    elif args.load_checkpoint is not None:
        model.load_weights(args.load_checkpoint)

    print(model.build_graph().summary(
        expand_nested=True))  # Print the Model summary

    # Training the model
    model.compile()
    model.fit(data, callbacks=callbacks)

    # Testing the model
    model.test(data, callbacks=callbacks)


if __name__ == "__main__":
    main()
