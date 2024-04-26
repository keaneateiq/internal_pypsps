"""Util functions for EIQ-specific implementation."""
from typing import Any, Union

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import tensorflow as tf
from matplotlib import pyplot as plt

import wandb
from internal_pypsps.external.pypsps import inference, utils

tfk = tf.keras

PREDICTED_OUTCOME = "predicted_outcome"
PROPENSITY_SCORE = "propensity_score"
UTE = "ute"


def prepare_data(
    X: Union[pd.DataFrame, np.ndarray],
    treatment: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray] = None,
) -> tuple:
    """Prepare data for training and prediction.

    Args:
        X: Feature Dataframe or Array
        treatment: Treatment Series or Array, should be binary
        y: Outcome Series or Array

    Return:
        input_data: List of input and treatment
        output_data: Array of stacked output and treatment
    """
    if (
        isinstance(X, np.ndarray)
        and isinstance(treatment, np.ndarray)
        and isinstance(y, np.ndarray)
    ):
        input_data = [X, treatment]
        output_data = np.hstack([y, treatment]) if y is not None else None
    else:
        input_data = [np.array(X.values).astype("float32"), treatment.values]
        output_data = (
            np.vstack([y.values, treatment.values]).T if y is not None else None
        )
    return input_data, output_data


def _eval_propensity(y_true, y_score):
    """Evaluate propensity score and visualize results.

    Args:
        y_true: True treatment values
        y_score: Predicted propensity scores
    """
    # TODO: Replace with curve estimator from ml-foundation
    y_comb = pd.DataFrame({"treatment": y_true, "propensity_score": y_score})
    plt.subplots(figsize=(10, 4))
    sns.displot(data=y_comb, x="propensity_score", hue="treatment")
    wandb.log(
        {
            "propensity_score_vs_treatment": wandb.Image(
                plt, caption="Propensity Score vs Treatment"
            )
        }
    )
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)
    sklearn.metrics.PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    auc_precision_recall = sklearn.metrics.auc(recall, precision)

    plt.title(f"Precision Recall Curve (AUC: {auc_precision_recall:.2f})")
    plt.grid()
    wandb.log(
        {"precision_recall_curve": wandb.Image(plt, caption="Precision Recall Curve")}
    )


def visualize_results(
    model: tf.keras.Model,
    X: Union[pd.DataFrame, np.ndarray],
    treatment: Union[pd.Series, np.ndarray],
    y: Union[pd.Series, np.ndarray],
):
    """Visualize the results."""
    # prepare inputs and outputs
    input_data, output_data = prepare_data(X, treatment, y)

    # inference
    y_pred = model.predict(input_data)
    outcome_pred, _, weights, propensity_score = utils.split_y_pred(y_pred)

    # evaluate propensity and show precision-call curve
    _eval_propensity(output_data[:, 1], propensity_score.ravel())

    # get the weights of all states
    weights_df = pd.DataFrame(
        weights, columns=["state" + str(i) for i in range(weights.shape[1])]
    )

    # show total weights of the states
    plt.subplots(figsize=(10, 4))
    # TODO: add a mean(weights_df.iloc[:, i] * treatment) for each i (barchart)
    weights_df.sum().plot.barh()
    wandb.log({"weights": wandb.Image(plt, caption="weights")})

    # show mean weights of the states for treatment = 1
    plt.subplots(figsize=(10, 4))
    if treatment.ndim == 1:
        treatment = treatment[:, np.newaxis]
    np.mean(weights_df * treatment).plot.barh()
    wandb.log(
        {
            "mean state weight for treatment = 1": wandb.Image(
                plt, caption="mean state weight for treatment = 1"
            )
        }
    )

    # show heatmap of the states
    plt.subplots(figsize=(10, 4))
    np.random.seed(42)  # sample 1000 units
    weights_df = weights_df.iloc[
        sorted(np.random.choice(range(weights_df.shape[0]), 500, replace=False)), :
    ]
    sns.heatmap(weights_df.round(2))
    wandb.log({"heatmap": wandb.Image(plt, caption="heatmap")})

    # show pairplot
    y_df = pd.DataFrame(
        outcome_pred,
        columns=["y_pred" + str(i) for i in range(outcome_pred.shape[1])],
    )
    y_df["outcome"] = output_data[:, 0]
    plt.subplots(figsize=(10, 4))
    sns.pairplot(y_df)
    wandb.log({"pairplot": wandb.Image(plt, caption="pairplot")})

    # show ute distribution
    ute = inference.predict_ute(model, input_data[0])
    _, axe = plt.subplots(figsize=(10, 4))
    sns.displot(ute, ax=axe)
    plt.axvline(
        x=output_data[output_data[:, 1] == 1, 0].mean()
        - output_data[output_data[:, 1] == 0, 0].mean(),
        ymax=axe.get_ylim()[1],
        color="red",
        label="naive ate",
    )
    plt.axvline(x=np.mean(ute), ymax=axe.get_ylim()[1], color="blue", label="ate")
    plt.legend()
    wandb.log({"ute": wandb.Image(plt, caption="ute")})

    # show ute distribution without naive ate
    _, axe = plt.subplots(figsize=(10, 4))
    sns.displot(ute, ax=axe)
    plt.axvline(x=np.mean(ute), ymax=axe.get_ylim()[1], color="blue", label="ate")
    plt.legend()
    wandb.log({"ute distribution without naive ate": wandb.Image(plt, caption="ute")})

    # show cumulative ute distribution
    plt.subplots(figsize=(10, 4))
    sns.ecdfplot(ute)
    plt.grid()
    wandb.log({"ute_cumulative": wandb.Image(plt, caption="ute_cumulative")})


def predict_outcome_propensity_ute(
    model: tf.keras.Model, features: Any, treatment: Any
) -> Union[pd.DataFrame, np.ndarray]:
    """Predicts outcome, propensity score and ute for binary treatment.

    Args:
      model: a trained pypsps model.
      features: features (X) for the causal model. Often a
        pd.DataFrame/np.ndarray, but can also be non-standard
        data structure as long as the pypsps model can use it as
        input to model.predict([features, ...]).

    Returns:
      A pd.DataFrame (if features is a DataFrame) or a np.ndarray of same number of
      rows as features with outcome and propensity_score as columns
    """
    # get ute
    weighted_ute = inference.predict_ute(model, features)
    weighted_ute = weighted_ute[:, np.newaxis]

    # get outcome and propensity score
    y_pred = inference.predict_counterfactual(model, features, treatment)
    outcome_pred, _, weights, propensity = utils.split_y_pred(y_pred)
    weighted_y = (weights * outcome_pred).sum(axis=1)
    weighted_y = weighted_y[:, np.newaxis]

    # concat
    results = np.concatenate([weighted_y, propensity, weighted_ute], axis=1)
    if isinstance(features, pd.DataFrame):
        return pd.DataFrame(
            results,
            index=features.index,
            columns=[PREDICTED_OUTCOME, PROPENSITY_SCORE, UTE],
        )
    return results
