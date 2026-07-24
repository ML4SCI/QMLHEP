from qml_hep_lhc.models.base_model import BaseModel


class QCNN(BaseModel):
    """
	General Quantum Convolutional Neural Network
	"""
    def __init__(self, data_config, args=None):
        super(QCNN, self).__init__(args)
        self.args = vars(args) if args is not None else {}

        # Data config
        self.input_dim = data_config["input_dims"]
        self.cluster_state = self.args.get("cluster_state", False)
        self.fm_class = self.args.get("feature_map", 'AngleMap')
        self.ansatz_class = self.args.get("ansatz", 'Chen')
        self.n_layers = self.args.get("n_layers", 1)
        self.n_qubits = self.args.get("n_qubits", 1)
        self.sparse = self.args.get("sparse", False)
        self.drc = self.args.get("drc", False)
        self.num_classes = len(data_config["mapping"])

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--cluster-state",
                            action="store_true",
                            default=False)
        parser.add_argument("--feature-map",
                            "-fm",
                            type=str,
                            default='AngleMap')
        parser.add_argument("--ansatz", type=str, default='Chen')
        parser.add_argument("--n-layers", type=int, default=1)
        parser.add_argument("--drc", action="store_true", default=False)
        parser.add_argument("--n-qubits", type=int, default=1)
        parser.add_argument("--sparse", action="store_true", default=False)
        parser.add_argument('--num-conv-layers', type=int, default=1)
        parser.add_argument('--conv-dims', action=ParseAction, default=[2])
        parser.add_argument('--num-fc-layers', type=int, default=1)
        parser.add_argument('--fc-dims', action=ParseAction, default=[128])
        parser.add_argument('--num-qconv-layers', type=int, default=1)
        parser.add_argument('--qconv-dims', action=ParseAction, default=[1])
        return parser
