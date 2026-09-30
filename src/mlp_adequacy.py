from pathlib import Path

import numpy as np

from src.learned_adequacy import (
    Standardizer,
)


class MLPAdequacyModel:
    """
    Small nonlinear Level-3 adequacy model.

    Input:
        agent-visible structured features only

    Output:
        risk score for:

            P-like score(Inadequate)

    IMPORTANT:
        Because class-weighted BCE is used,
        the output should currently be interpreted
        as a risk score, NOT a calibrated probability.

    Architecture:

        input
          ↓
        Linear
          ↓
        ReLU
          ↓
        Linear
          ↓
        ReLU
          ↓
        Linear
          ↓
        Sigmoid
    """

    def __init__(
        self,
        feature_names,
        hidden_1=32,
        hidden_2=16,
        threshold=0.5,
        seed=42,
    ):
        self.feature_names = list(
            feature_names
        )

        self.hidden_1 = int(
            hidden_1
        )

        self.hidden_2 = int(
            hidden_2
        )

        self.threshold = float(
            threshold
        )

        self.seed = int(
            seed
        )

        self.standardizer = (
            Standardizer()
        )

        self.parameters = None

    # ==================================================
    # Activations
    # ==================================================

    @staticmethod
    def _relu(
        x,
    ):
        return np.maximum(
            x,
            0.0,
        )

    @staticmethod
    def _relu_grad(
        x,
    ):
        return (
            x > 0.0
        ).astype(
            np.float64
        )

    @staticmethod
    def _sigmoid(
        x,
    ):
        x = np.clip(
            x,
            -50.0,
            50.0,
        )

        return (
            1.0
            /
            (
                1.0
                + np.exp(-x)
            )
        )

    # ==================================================
    # Initialization
    # ==================================================

    def _initialize(
        self,
        input_dim,
    ):
        rng = np.random.default_rng(
            self.seed
        )

        # He initialization for ReLU layers.
        w1 = (
            rng.standard_normal(
                (
                    input_dim,
                    self.hidden_1,
                )
            )
            *
            np.sqrt(
                2.0
                / input_dim
            )
        )

        b1 = np.zeros(
            self.hidden_1,
            dtype=np.float64,
        )

        w2 = (
            rng.standard_normal(
                (
                    self.hidden_1,
                    self.hidden_2,
                )
            )
            *
            np.sqrt(
                2.0
                / self.hidden_1
            )
        )

        b2 = np.zeros(
            self.hidden_2,
            dtype=np.float64,
        )

        # Smaller output initialization.
        w3 = (
            rng.standard_normal(
                (
                    self.hidden_2,
                    1,
                )
            )
            *
            np.sqrt(
                1.0
                / self.hidden_2
            )
        )

        b3 = np.zeros(
            1,
            dtype=np.float64,
        )

        self.parameters = {
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "w3": w3,
            "b3": b3,
        }

    # ==================================================
    # Forward
    # ==================================================

    def _forward(
        self,
        x,
    ):
        if self.parameters is None:
            raise RuntimeError(
                "Model parameters are not initialized."
            )

        w1 = self.parameters[
            "w1"
        ]

        b1 = self.parameters[
            "b1"
        ]

        w2 = self.parameters[
            "w2"
        ]

        b2 = self.parameters[
            "b2"
        ]

        w3 = self.parameters[
            "w3"
        ]

        b3 = self.parameters[
            "b3"
        ]

        z1 = (
            x @ w1
            + b1
        )

        a1 = self._relu(
            z1
        )

        z2 = (
            a1 @ w2
            + b2
        )

        a2 = self._relu(
            z2
        )

        logits = (
            a2 @ w3
            + b3
        )

        probabilities = (
            self._sigmoid(
                logits
            )
        )

        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2,
            "logits": logits,
        }

        return (
            probabilities,
            cache,
        )

    # ==================================================
    # Loss
    # ==================================================

    def _loss(
        self,
        probabilities,
        y,
        sample_weights,
        l2,
    ):
        epsilon = 1e-12

        probabilities = np.clip(
            probabilities,
            epsilon,
            1.0 - epsilon,
        )

        y_column = y.reshape(
            -1,
            1,
        )

        bce = -(
            y_column
            * np.log(
                probabilities
            )
            +
            (
                1.0
                - y_column
            )
            * np.log(
                1.0
                - probabilities
            )
        )

        weighted_bce = (
            sample_weights
            * bce
        )

        data_loss = (
            weighted_bce.sum()
            / sample_weights.sum()
        )

        w1 = self.parameters[
            "w1"
        ]

        w2 = self.parameters[
            "w2"
        ]

        w3 = self.parameters[
            "w3"
        ]

        regularization = (
            0.5
            * l2
            * (
                np.sum(
                    w1 * w1
                )
                +
                np.sum(
                    w2 * w2
                )
                +
                np.sum(
                    w3 * w3
                )
            )
        )

        return float(
            data_loss
            + regularization
        )

    # ==================================================
    # Backward
    # ==================================================

    def _backward(
        self,
        probabilities,
        y,
        sample_weights,
        cache,
        l2,
    ):
        x = cache[
            "x"
        ]

        z1 = cache[
            "z1"
        ]

        a1 = cache[
            "a1"
        ]

        z2 = cache[
            "z2"
        ]

        a2 = cache[
            "a2"
        ]

        w2 = self.parameters[
            "w2"
        ]

        w3 = self.parameters[
            "w3"
        ]

        y_column = y.reshape(
            -1,
            1,
        )

        normalizer = (
            sample_weights.sum()
        )

        # BCE + sigmoid derivative simplifies to:
        #
        #     probability - y
        #
        d_logits = (
            sample_weights
            * (
                probabilities
                - y_column
            )
            / normalizer
        )

        grad_w3 = (
            a2.T
            @ d_logits
            +
            l2
            * self.parameters[
                "w3"
            ]
        )

        grad_b3 = (
            d_logits.sum(
                axis=0
            )
        )

        d_a2 = (
            d_logits
            @ w3.T
        )

        d_z2 = (
            d_a2
            * self._relu_grad(
                z2
            )
        )

        grad_w2 = (
            a1.T
            @ d_z2
            +
            l2
            * self.parameters[
                "w2"
            ]
        )

        grad_b2 = (
            d_z2.sum(
                axis=0
            )
        )

        d_a1 = (
            d_z2
            @ w2.T
        )

        d_z1 = (
            d_a1
            * self._relu_grad(
                z1
            )
        )

        grad_w1 = (
            x.T
            @ d_z1
            +
            l2
            * self.parameters[
                "w1"
            ]
        )

        grad_b1 = (
            d_z1.sum(
                axis=0
            )
        )

        return {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
            "w3": grad_w3,
            "b3": grad_b3,
        }

    # ==================================================
    # Training
    # ==================================================

    def fit(
        self,
        x,
        y,
        epochs=800,
        learning_rate=0.003,
        l2=1e-4,
        positive_weight=None,
        patience=80,
        min_delta=1e-7,
        verbose=False,
    ):
        """
        Full-batch Adam optimization.

        IMPORTANT:
            early stopping here monitors TRAINING LOSS,
            not validation labels.

        Validation is reserved for:
            - model comparison
            - threshold analysis
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
                "x and y lengths differ."
            )

        if (
            x.shape[1]
            != len(
                self.feature_names
            )
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
                "No positive training samples."
            )

        if negatives == 0:
            raise ValueError(
                "No negative training samples."
            )

        if positive_weight is None:

            positive_weight = (
                negatives
                / positives
            )

        positive_weight = float(
            positive_weight
        )

        # ----------------------------------------------
        # Standardize using TRAIN ONLY
        # ----------------------------------------------

        x_scaled = (
            self.standardizer
            .fit_transform(
                x
            )
        )

        self._initialize(
            input_dim=
                x_scaled.shape[1]
        )

        sample_weights = np.where(
            y.reshape(
                -1,
                1,
            )
            == 1.0,

            positive_weight,

            1.0,
        )

        # ==============================================
        # Adam state
        # ==============================================

        first_moment = {}

        second_moment = {}

        for name, parameter in (
            self.parameters.items()
        ):

            first_moment[name] = (
                np.zeros_like(
                    parameter
                )
            )

            second_moment[name] = (
                np.zeros_like(
                    parameter
                )
            )

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        best_loss = float(
            "inf"
        )

        best_parameters = None

        stale_epochs = 0

        final_loss = None

        # ==============================================
        # Optimization
        # ==============================================

        for epoch in range(
            1,
            epochs + 1,
        ):

            (
                probabilities,
                cache,
            ) = self._forward(
                x_scaled
            )

            loss = self._loss(
                probabilities=
                    probabilities,

                y=y,

                sample_weights=
                    sample_weights,

                l2=l2,
            )

            gradients = (
                self._backward(
                    probabilities=
                        probabilities,

                    y=y,

                    sample_weights=
                        sample_weights,

                    cache=cache,

                    l2=l2,
                )
            )

            # ------------------------------------------
            # Adam update
            # ------------------------------------------

            for name in (
                self.parameters
            ):

                gradient = (
                    gradients[
                        name
                    ]
                )

                first_moment[name] = (
                    beta1
                    * first_moment[
                        name
                    ]
                    +
                    (
                        1.0
                        - beta1
                    )
                    * gradient
                )

                second_moment[name] = (
                    beta2
                    * second_moment[
                        name
                    ]
                    +
                    (
                        1.0
                        - beta2
                    )
                    * (
                        gradient
                        * gradient
                    )
                )

                m_hat = (
                    first_moment[
                        name
                    ]
                    /
                    (
                        1.0
                        - beta1 ** epoch
                    )
                )

                v_hat = (
                    second_moment[
                        name
                    ]
                    /
                    (
                        1.0
                        - beta2 ** epoch
                    )
                )

                self.parameters[name] -= (
                    learning_rate
                    * m_hat
                    /
                    (
                        np.sqrt(
                            v_hat
                        )
                        + epsilon
                    )
                )

            final_loss = loss

            # ------------------------------------------
            # Train-loss early stopping
            # ------------------------------------------

            if (
                loss
                <
                best_loss
                - min_delta
            ):

                best_loss = loss

                best_parameters = {
                    name:
                        parameter.copy()

                    for (
                        name,
                        parameter,
                    ) in (
                        self.parameters
                        .items()
                    )
                }

                stale_epochs = 0

            else:

                stale_epochs += 1

            if verbose and (
                epoch == 1
                or epoch % 50 == 0
            ):

                print(
                    f"epoch={epoch:4d} "
                    f"loss={loss:.6f}"
                )

            if (
                stale_epochs
                >= patience
            ):

                if verbose:

                    print(
                        "Early stopping at "
                        f"epoch={epoch}"
                    )

                break

        # Restore lowest train-loss parameters.
        if best_parameters is not None:

            self.parameters = (
                best_parameters
            )

        return {
            "epochs":
                epoch,

            "final_loss":
                float(
                    final_loss
                ),

            "best_loss":
                float(
                    best_loss
                ),

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
        if self.parameters is None:

            raise RuntimeError(
                "Model has not been fitted."
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        x_scaled = (
            self.standardizer
            .transform(
                x
            )
        )

        probabilities, _ = (
            self._forward(
                x_scaled
            )
        )

        return (
            probabilities[
                :,
                0,
            ]
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
            self.predict_proba(
                x
            )
        )

        return (
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

    def predict_feature_dict(
        self,
        feature_dict,
    ):
        row = [
            feature_dict[
                name
            ]
            for name
            in self.feature_names
        ]

        x = np.asarray(
            [
                row
            ],
            dtype=np.float64,
        )

        return float(
            self.predict_proba(
                x
            )[0]
        )

    # ==================================================
    # Save / load
    # ==================================================

    def save(
        self,
        path,
    ):
        if self.parameters is None:

            raise RuntimeError(
                "Cannot save unfitted model."
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,

            feature_names=
                np.asarray(
                    self.feature_names,
                    dtype=str,
                ),

            hidden_1=
                np.asarray(
                    self.hidden_1,
                    dtype=np.int64,
                ),

            hidden_2=
                np.asarray(
                    self.hidden_2,
                    dtype=np.int64,
                ),

            threshold=
                np.asarray(
                    self.threshold,
                    dtype=np.float64,
                ),

            seed=
                np.asarray(
                    self.seed,
                    dtype=np.int64,
                ),

            mean=
                self.standardizer.mean,

            std=
                self.standardizer.std,

            w1=
                self.parameters[
                    "w1"
                ],

            b1=
                self.parameters[
                    "b1"
                ],

            w2=
                self.parameters[
                    "w2"
                ],

            b2=
                self.parameters[
                    "b2"
                ],

            w3=
                self.parameters[
                    "w3"
                ],

            b3=
                self.parameters[
                    "b3"
                ],
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

            hidden_1=
                int(
                    data[
                        "hidden_1"
                    ]
                ),

            hidden_2=
                int(
                    data[
                        "hidden_2"
                    ]
                ),

            threshold=
                float(
                    data[
                        "threshold"
                    ]
                ),

            seed=
                int(
                    data[
                        "seed"
                    ]
                ),
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

        model.parameters = {
            "w1":
                data[
                    "w1"
                ].astype(
                    np.float64
                ),

            "b1":
                data[
                    "b1"
                ].astype(
                    np.float64
                ),

            "w2":
                data[
                    "w2"
                ].astype(
                    np.float64
                ),

            "b2":
                data[
                    "b2"
                ].astype(
                    np.float64
                ),

            "w3":
                data[
                    "w3"
                ].astype(
                    np.float64
                ),

            "b3":
                data[
                    "b3"
                ].astype(
                    np.float64
                ),
        }

        return model