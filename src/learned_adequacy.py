from pathlib import Path

import numpy as np


class Standardizer:
    """
    Train-set-only feature standardization.

    IMPORTANT:
        fit() must only be called on training data.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(
        self,
        x: np.ndarray,
    ):
        if x.ndim != 2:
            raise ValueError(
                "x must be 2-D."
            )

        self.mean = x.mean(
            axis=0
        )

        self.std = x.std(
            axis=0
        )

        # Avoid division by zero for
        # constant features.
        self.std = np.where(
            self.std < 1e-8,
            1.0,
            self.std,
        )

        return self

    def transform(
        self,
        x: np.ndarray,
    ):
        if (
            self.mean is None
            or self.std is None
        ):
            raise RuntimeError(
                "Standardizer has not been fitted."
            )

        return (
            (x - self.mean)
            / self.std
        )

    def fit_transform(
        self,
        x: np.ndarray,
    ):
        self.fit(x)

        return self.transform(x)


class LogisticAdequacyModel:
    """
    Learned Level-3 adequacy baseline.

    Input:
        agent-visible features only

    Output:
        P(Inadequate)

    This is deliberately simple and interpretable.

    It provides a learned baseline before moving
    to nonlinear MLP / latent representations.
    """

    def __init__(
        self,
        feature_names,
        threshold=0.5,
    ):
        self.feature_names = list(
            feature_names
        )

        self.threshold = float(
            threshold
        )

        self.standardizer = (
            Standardizer()
        )

        self.weights = None
        self.bias = 0.0

    # ==================================================
    # Math
    # ==================================================

    @staticmethod
    def _sigmoid(
        z,
    ):
        z = np.clip(
            z,
            -50.0,
            50.0,
        )

        return (
            1.0
            /
            (
                1.0
                + np.exp(-z)
            )
        )

    # ==================================================
    # Training
    # ==================================================

    def fit(
        self,
        x,
        y,
        epochs=1500,
        learning_rate=0.05,
        l2=1e-4,
        positive_weight=None,
        tolerance=1e-8,
        patience=80,
        verbose=False,
    ):
        """
        Weighted binary logistic regression.

        Label:
            1 = inadequate
            0 = adequate

        positive_weight compensates for the
        smaller inadequate class.
        """

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        y = np.asarray(
            y,
            dtype=np.float64,
        )

        if x.ndim != 2:
            raise ValueError(
                "x must be 2-D."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be 1-D."
            )

        if len(x) != len(y):
            raise ValueError(
                "x and y have different lengths."
            )

        if (
            x.shape[1]
            != len(self.feature_names)
        ):
            raise ValueError(
                "Feature count mismatch."
            )

        positives = int(
            y.sum()
        )

        negatives = (
            len(y)
            - positives
        )

        if positives == 0:
            raise ValueError(
                "Training set contains "
                "no positive samples."
            )

        if negatives == 0:
            raise ValueError(
                "Training set contains "
                "no negative samples."
            )

        if positive_weight is None:

            positive_weight = (
                negatives
                / positives
            )

        positive_weight = float(
            positive_weight
        )

        x_scaled = (
            self.standardizer
            .fit_transform(x)
        )

        feature_count = (
            x_scaled.shape[1]
        )

        # Zero initialization is deterministic
        # and valid for logistic regression.
        self.weights = np.zeros(
            feature_count,
            dtype=np.float64,
        )

        self.bias = 0.0

        sample_weights = np.where(
            y == 1.0,
            positive_weight,
            1.0,
        )

        normalizer = (
            sample_weights.sum()
        )

        previous_loss = None
        stable_epochs = 0

        for epoch in range(
            1,
            epochs + 1,
        ):

            logits = (
                x_scaled
                @ self.weights
                + self.bias
            )

            probabilities = (
                self._sigmoid(
                    logits
                )
            )

            error = (
                sample_weights
                * (
                    probabilities
                    - y
                )
            )

            grad_weights = (
                (
                    x_scaled.T
                    @ error
                )
                / normalizer
                +
                l2
                * self.weights
            )

            grad_bias = (
                error.sum()
                / normalizer
            )

            self.weights -= (
                learning_rate
                * grad_weights
            )

            self.bias -= (
                learning_rate
                * grad_bias
            )

            # ------------------------------------------
            # Weighted BCE loss
            # ------------------------------------------

            epsilon = 1e-12

            probabilities = np.clip(
                probabilities,
                epsilon,
                1.0 - epsilon,
            )

            bce = -(
                y
                * np.log(
                    probabilities
                )
                +
                (1.0 - y)
                * np.log(
                    1.0
                    - probabilities
                )
            )

            loss = (
                (
                    sample_weights
                    * bce
                ).sum()
                / normalizer
                +
                0.5
                * l2
                * np.dot(
                    self.weights,
                    self.weights,
                )
            )

            if verbose and (
                epoch == 1
                or epoch % 100 == 0
            ):

                print(
                    f"epoch={epoch:4d} "
                    f"loss={loss:.6f}"
                )

            # ------------------------------------------
            # Early stopping
            # ------------------------------------------

            if previous_loss is not None:

                improvement = abs(
                    previous_loss
                    - loss
                )

                if (
                    improvement
                    < tolerance
                ):
                    stable_epochs += 1

                else:
                    stable_epochs = 0

                if (
                    stable_epochs
                    >= patience
                ):
                    if verbose:

                        print(
                            "Early stopping at "
                            f"epoch={epoch}"
                        )

                    break

            previous_loss = loss

        return {
            "epochs":
                epoch,

            "final_loss":
                float(loss),

            "positive_weight":
                positive_weight,
        }

    # ==================================================
    # Inference
    # ==================================================

    def predict_proba(
        self,
        x,
    ):
        if self.weights is None:
            raise RuntimeError(
                "Model has not been fitted."
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        x_scaled = (
            self.standardizer
            .transform(x)
        )

        logits = (
            x_scaled
            @ self.weights
            + self.bias
        )

        return (
            self._sigmoid(
                logits
            )
        )

    def predict(
        self,
        x,
        threshold=None,
    ):
        if threshold is None:
            threshold = (
                self.threshold
            )

        probabilities = (
            self.predict_proba(x)
        )

        return (
            probabilities
            >= threshold
        ).astype(np.int64)

    def predict_feature_dict(
        self,
        feature_dict,
    ):
        """
        Future online Level-3 interface.

        Takes one agent-visible feature dictionary
        and returns P(Inadequate).
        """

        row = [
            feature_dict[name]
            for name
            in self.feature_names
        ]

        x = np.asarray(
            [row],
            dtype=np.float64,
        )

        probability = (
            self.predict_proba(x)[0]
        )

        return float(
            probability
        )

    # ==================================================
    # Persistence
    # ==================================================

    def save(
        self,
        path,
    ):
        if self.weights is None:
            raise RuntimeError(
                "Cannot save an unfitted model."
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,

            feature_names=np.asarray(
                self.feature_names,
                dtype=str,
            ),

            threshold=np.asarray(
                self.threshold,
                dtype=np.float64,
            ),

            weights=self.weights,

            bias=np.asarray(
                self.bias,
                dtype=np.float64,
            ),

            mean=self.standardizer.mean,

            std=self.standardizer.std,
        )

    @classmethod
    def load(
        cls,
        path,
    ):
        data = np.load(
            path,
            allow_pickle=False,
        )

        feature_names = [
            str(value)
            for value
            in data[
                "feature_names"
            ].tolist()
        ]

        model = cls(
            feature_names=
                feature_names,

            threshold=float(
                data["threshold"]
            ),
        )

        model.weights = (
            data[
                "weights"
            ].astype(
                np.float64
            )
        )

        model.bias = float(
            data["bias"]
        )

        model.standardizer.mean = (
            data[
                "mean"
            ].astype(
                np.float64
            )
        )

        model.standardizer.std = (
            data[
                "std"
            ].astype(
                np.float64
            )
        )

        return model