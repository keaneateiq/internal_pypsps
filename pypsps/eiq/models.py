"""Model class for pypsps that is compatible and can be consumed by EIQ model wrapper."""

from typing import List
from abc import ABCMeta
import numpy as np
import tensorflow as tf
import pandas as pd
import pypress
import pypress.keras.layers
import pypress.keras.regularizers
from pypsps.keras import losses, metrics, models
from pypsps import inference
from typing import Union

tfk = tf.keras

class BaseLearner(metaclass=ABCMeta):
    """Base class for causal learners."""
    @classmethod
    def fit(self, X, treatment, y):
        pass

    @classmethod
    def predict(self, X, treatment):
        pass

class PSPSLearner(BaseLearner):
    """PSPS Learner class."""
    def __init__(
        self,
        n_states: int,
        n_features: int,
        alpha: float = 1.0,
        epochs: int = 50,
        validation_split: float = 0.2,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        outcome_loss_weight: float = 0.01,
        df_regularizer_l1: float = 10.0
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
        
        self.model = models.build_toy_model(
            n_states=self.n_states, n_features=self.n_features, alpha=self.alpha
        )
        
    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series, **kwargs):
        """Fit the model.
        
        Args:
            X: Feature DataFrame
            treatment: Treatment Series, should be binary
            y: Outcome Series
        """
       # prepare inputs and outputs
        input_data = [np.array(X.values).astype("float32"), treatment.values]
        output_data = np.vstack([y.values, treatment.values]).T

        # build loss function
        psps_outcome_loss = losses.OutcomeLoss(loss=losses.NegloglikNormal(reduction="none"), reduction="auto")
        psps_treat_loss = losses.TreatmentLoss(loss=tf.keras.losses.BinaryCrossentropy(reduction="none"), reduction="auto")
        psps_causal_loss = losses.CausalLoss(outcome_loss=psps_outcome_loss,
                                  treatment_loss=psps_treat_loss,
                                  alpha=self.alpha,
                                  outcome_loss_weight=self.outcome_loss_weight,
                                  predictive_states_regularizer=pypress.keras.regularizers.DegreesOfFreedom(l1 = self.df_regularizer_l1, df= self.n_states - 1),
                                  reduction="auto")

        # compile model
        self.model.compile(loss=psps_causal_loss,
              optimizer=tfk.optimizers.Nadam(learning_rate=self.learning_rate),
              metrics=[metrics.propensity_score_crossentropy])

        # fit
        history = self.model.fit(input_data, output_data, epochs=self.epochs, batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=models.recommended_callbacks(), **kwargs)
        
        return self

    def predict(self, X: pd.DataFrame, treatment: pd.Series) -> Union[pd.DataFrame, np.ndarray]:
        """Predict treatment effect.
        
        Args:
            X: Feature Dataframe
            treatment: Treatment Series, should be binary

        Return:
            Outcome and propensity score for each row in X
        """
        X = np.array(X.values).astype("float32")
        return inference.predict_outcome_propensity_ute(self.model, X, treatment)