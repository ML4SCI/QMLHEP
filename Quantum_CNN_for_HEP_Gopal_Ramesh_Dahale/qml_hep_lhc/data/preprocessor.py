import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from numba import njit, prange
from tensorflow import image
from tensorflow.keras.utils import to_categorical
from qml_hep_lhc.utils import ParseAction


class DataPreprocessor():
    """
    Data Preprocessing Module
    """
    def __init__(self, args=None) -> None:

        # Load the data and arguments
        self.args = args if args is not None else {}

        # Parse args
        self._labels_to_categorical = self.args.get("labels_to_categorical",
                                                    False)
        self._normalize = self.args.get("normalize", False)
        self._standardize = self.args.get("standardize", False)
        self._min_max = self.args.get("min_max", False)
        self._resize = self.args.get("resize", None)
        self._binary_data = self.args.get("binary_data", None)
        self._is_binary_data = self.args.get("is_binary_data", False)
        self._pca = self.args.get("pca", None)
        self._graph_conv = self.args.get("graph_conv", False)
        self._center_crop = self.args.get("center_crop", None)
        self._to_rgb = self.args.get("to_rgb", False)
        self._power_transform = self.args.get("power_transform", False)
        self._hinge_labels = self.args.get("hinge_labels", False)

        if self._is_binary_data:
            self._binary_data = None

    def standardize(self, x_train, x_test):
        """
        Standardize features by removing the mean and scaling to unit variance.
        """
        print("Standardizing data...")

        img_size = self.dims
        std_slr = StandardScaler()

        x_train = x_train.reshape(-1, np.prod(img_size))
        x_test = x_test.reshape(-1, np.prod(img_size))

        x_train = std_slr.fit_transform(x_train)
        x_test = std_slr.transform(x_test)

        x_train = x_train.reshape([-1] + list(img_size))
        x_test = x_test.reshape([-1] + list(img_size))

        return x_train, x_test

    def normalize_data(self, x_train, x_test):
        """
        Scale input vectors individually to unit norm (vector length).
        """
        print("Normalizing data...")

        img_size = self.dims
        normalizer = Normalizer()

        x_train = x_train.reshape(-1, np.prod(img_size))
        x_test = x_test.reshape(-1, np.prod(img_size))

        x_train = normalizer.fit_transform(x_train)
        x_test = normalizer.transform(x_test)

        x_train = x_train.reshape([-1] + list(img_size))
        x_test = x_test.reshape([-1] + list(img_size))

        return x_train, x_test

    def min_max_scale(self, x_train, x_test):
        print("Min-max scaling...")

        img_size = self.dims
        min_max_slr = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2))

        x_train = x_train.reshape(-1, np.prod(img_size))
        x_test = x_test.reshape(-1, np.prod(img_size))

        x_train = min_max_slr.fit_transform(x_train)
        x_test = min_max_slr.transform(x_test)

        x_train = x_train.reshape([-1] + list(img_size))
        x_test = x_test.reshape([-1] + list(img_size))

        return x_train, x_test

    def resize(self, x):
        """
        It resizes the training and testing data to the size specified in the constructor
        """
        print("Resizing data...")
        x = image.resize(x, self._resize).numpy()
        self.dims = x.shape[1:]
        return x

    def labels_to_categorical(self, y):
        """
        It converts the labels to categorical data
        """
        print("Converting labels to categorical...")

        y = to_categorical(y, num_classes=len(self.mapping))
        self.output_dims = (len(self.mapping), )
        return y

    def hinge_labels(self, y):
        print("Hinge labels...")
        return 2 * y - 1

    def binary_data(self, x, y):
        """
        It takes the data and filters it so that only the data that contains binary classes
        """
        print("Binarizing data...")

        if self._is_binary_data is False:

            # Get the binary classes
            d1 = self._binary_data[0]
            d2 = self._binary_data[1]

            # Extract binary data
            x, y = binary_filter(d1, d2, x, y)
        return x, y

    def pca(self, x_train, x_test, n_components=16):
        """
        Performs Principal component analysis (PCA) on the data.

        Args:
          n_components: Number of components to keep. If n_components is not set all components are
        kept:. Defaults to 16
        """
        print("Performing PCA on data...")

        sq_root = int(np.sqrt(n_components))
        assert sq_root * sq_root == n_components, "Number of components must be a square"

        pca_obj = PCA(n_components)

        x_train = x_train.reshape(-1, np.prod(self.dims))
        x_test = x_test.reshape(-1, np.prod(self.dims))

        x_train = pca_obj.fit_transform(x_train)
        cumsum = np.cumsum(pca_obj.explained_variance_ratio_ * 100)[-1]
        print("Cumulative sum on train :", cumsum)

        x_test = pca_obj.transform(x_test)
        cumsum = np.cumsum(pca_obj.explained_variance_ratio_ * 100)[-1]
        print("Cumulative sum on test:", cumsum)

        x_train = x_train.reshape(-1, sq_root, sq_root, 1)
        x_test = x_test.reshape(-1, sq_root, sq_root, 1)

        self.dims = (sq_root, sq_root, 1)
        return x_train, x_test

    def graph_convolution(self, x):
        print("Performing graph convolution...")
        m = self.dims[0]
        n = self.dims[1]
        N = m * n
        adj = np.zeros((N, N))
        sigma = np.pi

        # Create adjacency matrix
        @njit(parallel=True)
        def fill(adj, i):
            for j in prange(i, N):
                p1 = np.array([i // n, i % n])
                p2 = np.array([j // n, j % n])
                d = np.sqrt(np.sum(np.square(p1 - p2)))
                val = np.exp(-d / (sigma**2))
                adj[i][j] = val
                adj[j][i] = val

        def iterate(adj):
            for i in prange(N):
                fill(adj, i)

        iterate(adj)

        # Perfrom graph convolution

        x = [
            np.dot(adj, x[:, :, :, i].reshape(-1, N,
                                              1).T).T.reshape(-1, m, n, 1)
            for i in range(self.dims[2])
        ]
        x = np.concatenate(x, axis=3)
        return x

    def center_crop(self, x, fraction=0.2):
        print("Center cropping...")
        x = image.central_crop(x, fraction).numpy()
        self.dims = x.shape[1:]
        return x

    def power_transform(self, x_train, x_test):
        print("Performing power transform...")

        img_size = self.dims
        pt = PowerTransformer()

        x_train = x_train.reshape(-1, np.prod(img_size))
        x_test = x_test.reshape(-1, np.prod(img_size))

        x_train = pt.fit_transform(x_train)
        x_test = pt.transform(x_test)

        x_train = x_train.reshape([-1] + list(img_size))
        x_test = x_test.reshape([-1] + list(img_size))

        return x_train, x_test

    def process(self, x_train, y_train, x_test, y_test, config, classes):
        """
        Data processing pipeline.
        """

        self.dims = config['input_dims']
        self.output_dims = config['output_dims']
        self.mapping = config['mapping']
        self.classes = classes

        # Add new axis
        # For resizing we need to add one more axis
        if len(x_train.shape) == 3:
            x_train = x_train[..., np.newaxis]
        if len(x_test.shape) == 3:
            x_test = x_test[..., np.newaxis]

        if self._binary_data and len(self._binary_data) == 2:

            # Get the binary classes
            d1 = self._binary_data[0]
            d2 = self._binary_data[1]

            x_train, y_train = self.binary_data(x_train, y_train)
            x_test, y_test = self.binary_data(x_test, y_test)

            self.mapping = [0, 1]
            self.classes = [self.classes[d1], self.classes[d2]]

        if self._to_rgb:
            print("Converting to RGB...")
            x_train = np.repeat(x_train, 3, axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
            self.dims = x_train.shape[1:]

        if self._center_crop:
            x_train = self.center_crop(x_train, self._center_crop)
            x_test = self.center_crop(x_test, self._center_crop)

        if self._resize is not None and len(self._resize) == 2:
            x_train = self.resize(x_train)
            x_test = self.resize(x_test)

        if self._graph_conv:
            x_train = self.graph_convolution(x_train)
            x_test = self.graph_convolution(x_test)

        if self._power_transform:
            x_train, x_test = self.power_transform(x_train, x_test)
        if self._pca is not None:
            x_train, x_test = self.pca(x_train, x_test, self._pca)
        if self._standardize:
            x_train, x_test = self.standardize(x_train, x_test)
        if self._normalize:
            x_train, x_test = self.normalize_data(x_train, x_test)
        if self._min_max:
            x_train, x_test = self.min_max_scale(x_train, x_test)

        if self._hinge_labels:
            y_train = self.hinge_labels(y_train)
            y_test = self.hinge_labels(y_test)

        if self._labels_to_categorical:
            y_train = self.labels_to_categorical(y_train)
            y_test = self.labels_to_categorical(y_test)

        return x_train, y_train, x_test, y_test

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--labels-to-categorical",
                            "-to-cat",
                            action="store_true",
                            default=False)
        parser.add_argument("--normalize",
                            "-nz",
                            action="store_true",
                            default=False)
        parser.add_argument("--standardize",
                            "-std",
                            action="store_true",
                            default=False)
        parser.add_argument("--min-max",
                            "-mm",
                            action="store_true",
                            default=False)
        parser.add_argument("--resize",
                            "-rz",
                            action=ParseAction,
                            default=None)
        parser.add_argument("--binary-data",
                            "-bd",
                            action=ParseAction,
                            default=None)
        parser.add_argument("--pca", "-pca", type=int, default=None)
        parser.add_argument("--graph-conv",
                            "-gc",
                            action="store_true",
                            default=False)
        parser.add_argument("--center-crop", "-cc", type=float, default=None)
        parser.add_argument("--to-rgb",
                            "-rgb",
                            action="store_true",
                            default=False)
        parser.add_argument("--power-transform",
                            "-pt",
                            action="store_true",
                            default=False)
        parser.add_argument("--hinge-labels",
                            "-hl",
                            action="store_true",
                            default=False)
        return parser


def binary_filter(d1, d2, x, y):
    """
    It takes a dataset and two labels, and returns a dataset with only those two labels
    
    Args:
      d1: the first digit to filter for
      d2: the second digit to keep
      x: the data
      y: the labels
    
    Returns:
      the x and y values that are either d1 or d2.
    """
    keep = (y == d1) | (y == d2)
    x, y = x[keep], y[keep]
    y = (y == d1)
    return x, y
