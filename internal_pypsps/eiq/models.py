"""Model class for pypsps that is compatible and can be consumed by EIQ model wrapper."""

from abc import ABCMeta
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from wandb.integration import keras as wandb_keras

from internal_pypsps.external.pypsps.keras import models

from . import utils as eiq_utils

tfk = tf.keras


class BaseLearner(metaclass=ABCMeta):
    """Base class for causal learners."""

    def fit(self, X, treatment, y, **kwargs):
        """Fit the model."""

    def predict(self, X, treatment):
        """Predict."""


class PSPSLearner(BaseLearner):
    """PSPS Learner class."""

    # pylint: disable=R0902
    def __init__(
        self,
        n_states: int,
        n_features: int,
        alpha: float = 1.0,
        epochs: int = 2,
        validation_split: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        outcome_loss_weight: float = 0.01,
        df_regularizer_l1: float = 10.0,
        **kwargs
    ):
        """Initialize the PSPS Learner.

        Args:
            n_states: Number of predictive states
            n_features: Number of features
            alpha: Regularization parameter
            epochs: Number of epochs
            validation_split: Validation split
            batch_size: Batch size
            learning_rate: Learning rate
            outcome_loss_weight: Outcome loss weight
            df_regularizer_l1: L1 regularization parameter for the degrees of freedom
        """
        self.n_states = n_states
        self.n_features = n_features
        self.alpha = alpha
        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.outcome_loss_weight = outcome_loss_weight
        self.df_regularizer_l1 = df_regularizer_l1

        self.model = models.build_model_binary_normal(
            n_states=self.n_states,
            n_features=self.n_features,
            alpha=self.alpha,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series, **kwargs):
        """Fit the model.

        Args:
            X: Feature DataFrame
            treatment: Treatment Series, should be binary
            y: Outcome Series
        """
        # prepare inputs and outputs
        input_data, output_data = eiq_utils.prepare_data(X, treatment, y)

        # fit
        self.model.fit(
            input_data,
            output_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=models.recommended_callbacks() + [wandb_keras.WandbCallback()],
            **kwargs,
        )

        # visualize results
        eiq_utils.visualize_results(self.model, X, treatment, y)

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Predict treatment effect.

        Args:
            X: Feature Dataframe
            treatment: Treatment Series, should be binary

        Return:
            Outcome and propensity score for each row in X
        """
        (X, treatment), _ = eiq_utils.prepare_data(X, treatment)
        return eiq_utils.predict_outcome_propensity_ute(self.model, X, treatment)
