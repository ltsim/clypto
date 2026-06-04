import numbers as nb
import numpy as np

class LabelEncoder:
    """
    Encode categorical features as integer labels.
    Especially, it can encode a list of mixed types include integer, float, and string. Better than scikit-learn module.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    @staticmethod
    def set_y(y):
        if type(y) not in (list, tuple, np.ndarray):
            y = (y,)

        return y

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.
        """

        def safe_key(val):
            # Chuyển None -> 0, số -> 1, chuỗi -> 2, object khác -> 3
            if val is None:
                return (0, "")
            elif isinstance(val, nb.Number):
                return (1, val)
            elif isinstance(val, str):
                return (2, val)
            else:
                return (3, str(val))

        # self.unique_labels = sorted(set(y), key=lambda x: (isinstance(x, (int, float)), x))
        self.unique_labels = sorted(set(y), key=safe_key)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

        return self

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : list, tuple
            Labels to encode.

        Returns:
        --------
        encoded_labels : list
            Encoded integer labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")

        y = self.set_y(y)

        return [self.label_to_index[label] for label in y]

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : list, tuple
            Target values.

        Returns
        -------
        y : list
            Encoded labels.
        """
        y = self.set_y(y)

        self.fit(y)

        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : list, tuple
            Encoded integer labels.

        Returns:
        --------
        original_labels : list
            Original labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")

        y = self.set_y(y)

        return [
            self.unique_labels[i] if i in self.label_to_index.values() else "unknown"
            for i in y
        ]
