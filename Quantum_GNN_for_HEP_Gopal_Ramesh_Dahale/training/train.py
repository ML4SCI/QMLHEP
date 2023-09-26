"""Library file for executing training and evaluation on ogbg-molpcba."""

import os
from typing import Any, Dict, Iterable, Tuple, Optional

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.core
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax
import sklearn.metrics
import tensorflow as tf
from training.util import import_class
from qgnn_hep.data import input_pipeline
import wandb


def create_model(model_name: str, model_hparams: ml_collections.ConfigDict, deterministic: bool) -> nn.Module:
    model_class = import_class(f"qgnn_hep.models.{model_name}")
    return model_class(**model_hparams, deterministic=deterministic)


def create_optimizer(config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Creates an optimizer, as specified by the config."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate=config.learning_rate)
    if config.optimizer == "sgd":
        return optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


def binary_cross_entropy_with_mask(*, logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray):
    """Binary cross entropy loss for unnormalized logits, with masked elements."""
    assert logits.shape == labels.shape == mask.shape
    assert len(logits.shape) == 2

    # To prevent propagation of NaNs during grad().
    # We mask over the loss for invalid targets later.
    labels = jnp.where(mask, labels, -1)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = logits >= 0
    relu_logits = jnp.where(positive_logits, logits, 0)
    abs_logits = jnp.where(positive_logits, logits, -logits)
    return relu_logits - (logits * labels) + (jnp.log(1 + jnp.exp(-abs_logits)))


def predictions_match_labels(*, logits: jnp.ndarray, labels: jnp.ndarray, **kwargs) -> jnp.ndarray:
    """Returns a binary array indicating where predictions match the labels."""
    del kwargs  # Unused.
    preds = logits > 0
    return (preds == labels).astype(jnp.float32)


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}_{key}": val for key, val in result.items()}


@flax.struct.dataclass
class MeanAveragePrecision(metrics.CollectingMetric.from_outputs(("labels", "logits", "mask"))):
    """Computes the mean average precision (mAP) over different tasks."""

    def compute(self):
        # Matches the official OGB evaluation scheme for mean average precision.
        values = super().compute()
        labels = values["labels"]
        logits = values["logits"]
        mask = values["mask"]

        assert logits.shape == labels.shape == mask.shape
        assert len(logits.shape) == 2

        probs = jax.nn.sigmoid(logits)
        num_tasks = labels.shape[1]
        average_precisions = np.full(num_tasks, np.nan)

        for task in range(num_tasks):
            # AP is only defined when there is at least one negative data
            # and at least one positive data.
            is_labeled = mask[:, task]
            if len(np.unique(labels[is_labeled, task])) >= 2:
                average_precisions[task] = sklearn.metrics.average_precision_score(
                    labels[is_labeled, task], probs[is_labeled, task]
                )

        # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
        if np.isnan(average_precisions).all():
            return np.nan
        return np.nanmean(average_precisions)


@flax.struct.dataclass
class AUC(metrics.CollectingMetric.from_outputs(("labels", "logits", "mask"))):
    """Computes the ROC AUC Score"""

    def compute(self):
        values = super().compute()
        labels = values["labels"]
        logits = values["logits"]
        mask = values["mask"]

        assert logits.shape == labels.shape == mask.shape
        assert len(logits.shape) == 2

        # take first column of mask
        mask = mask[:, 0]

        # We mask over the AUC score for invalid targets.
        labels = jnp.argmax(labels, axis=-1)
        masked_labels = labels[mask]
        probs = jax.nn.softmax(logits)
        probs = probs[:, 1]
        masked_probs = probs[mask]

        return sklearn.metrics.roc_auc_score(masked_labels, masked_probs)


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):

    accuracy: metrics.Average.from_fun(predictions_match_labels)
    loss: metrics.Average.from_output("loss")
    mean_average_precision: MeanAveragePrecision
    auc: AUC


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):

    accuracy: metrics.Average.from_fun(predictions_match_labels)
    loss: metrics.Average.from_output("loss")


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Replaces the globals attribute with a constant feature for each graph."""
    return graphs._replace(globals=jnp.ones([graphs.n_node.shape[0], 1]))


def get_predicted_logits(
    state: train_state.TrainState, graphs: jraph.GraphsTuple, rngs: Optional[Dict[str, jnp.ndarray]]
) -> jnp.ndarray:
    """Get predicted logits from the network for input graphs."""
    pred_graphs = state.apply_fn(state.params, graphs, rngs=rngs)
    logits = pred_graphs.globals
    return logits


def get_valid_mask(labels: jnp.ndarray, graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Gets the binary mask indicating only valid labels and graphs."""
    # We have to ignore all NaN values - which indicate labels for which
    # the current graphs have no label.
    labels_mask = ~jnp.isnan(labels)

    # Since we have extra 'dummy' graphs in our batch due to padding, we want
    # to mask out any loss associated with the dummy graphs.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.
    graph_mask = jraph.get_graph_padding_mask(graphs)

    # Combine the mask over labels with the mask over graphs.
    return labels_mask & graph_mask[:, None]


@jax.jit
def train_step(
    state: train_state.TrainState, graphs: jraph.GraphsTuple, rngs: Dict[str, jnp.ndarray]
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params, graphs):
        curr_state = state.replace(params=params)

        # Extract labels.
        labels = graphs.globals
        labels = jax.nn.one_hot(labels, 2)

        # Replace the global feature for graph classification.
        graphs = replace_globals(graphs)

        # Compute logits and resulting loss.
        logits = get_predicted_logits(curr_state, graphs, rngs)
        mask = get_valid_mask(labels, graphs)
        loss = binary_cross_entropy_with_mask(logits=logits, labels=labels, mask=mask)
        mean_loss = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)

        return mean_loss, (loss, logits, labels, mask)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (loss, logits, labels, mask)), grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)

    metrics_update = TrainMetrics.single_from_model_output(loss=loss, logits=logits, labels=labels, mask=mask)
    return state, metrics_update


@jax.jit
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""

    # The target labels our model has to predict.
    labels = graphs.globals
    labels = jax.nn.one_hot(labels, 2)

    # Replace the global feature for graph classification.
    graphs = replace_globals(graphs)

    # Get predicted logits, and corresponding probabilities.
    logits = get_predicted_logits(state, graphs, rngs=None)

    # Get the mask for valid labels and graphs.
    mask = get_valid_mask(labels, graphs)

    # Compute the various metrics.
    loss = binary_cross_entropy_with_mask(logits=logits, labels=labels, mask=mask)

    return EvalMetrics.single_from_model_output(loss=loss, logits=logits, labels=labels, mask=mask)


def evaluate_model(
    state: train_state.TrainState, datasets: Dict[str, tf.data.Dataset], splits: Iterable[str]
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = None

        # Loop over graphs.
        for graphs in datasets[split].as_numpy_iterator():
            split_metrics_update = evaluate_step(state, graphs)

            # Update metrics.
            if split_metrics is None:
                split_metrics = split_metrics_update
                # logging.info("Updated split metrics for the first time")
            else:
                split_metrics = split_metrics.merge(split_metrics_update)
                # logging.info("Updated split metrics")
        eval_metrics[split] = split_metrics

    return eval_metrics  # pytype: disable=bad-return-type


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str, wandb_logging: bool) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
            config: Hyperparameter configuration for training and evaluation.
            workdir: Directory where the TensorBoard summaries are written to.

    Returns:
            The train state (which includes the `.params`).
    """
    # We only support single-host training.
    assert jax.process_count() == 1

    train_hparams = config.train_hparams
    model_hparams = config.model_hparams
    optimizer_hparams = config.optimizer_hparams

    # Create writer for logs.
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    # Initializing a Weights & Biases Run
    if wandb_logging:
        wandb.init(project="qgnn-hep", config=config.to_dict())

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    datasets = input_pipeline.get_datasets(
        train_hparams.dataset_name,
        train_hparams.dataset_config_name,
        train_hparams.batch_size,
        add_virtual_node=train_hparams.add_virtual_node,
        add_undirected_edges=train_hparams.add_undirected_edges,
        add_self_loops=train_hparams.add_self_loops,
    )
    train_iter = iter(datasets["train"])

    # Create and initialize the network.
    logging.info("Initializing network.")
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    init_graphs = next(datasets["train"].as_numpy_iterator())
    init_graphs = replace_globals(init_graphs)
    print("init_graphs", init_graphs.nodes.shape, init_graphs.edges.shape, init_graphs.globals.shape)

    # Get model class
    model_name = train_hparams.model_name
    init_net = create_model(model_name, model_hparams, deterministic=True)
    params = jax.jit(init_net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = create_optimizer(optimizer_hparams)

    # Create the training state.
    net = create_model(model_name, model_hparams, deterministic=False)
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # Set up checkpointing of the model.
    # The input pipeline cannot be checkpointed in its current form,
    # due to the use of stateful operations.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)

    if train_hparams.load_checkpoint:
        state = ckpt.restore_or_initialize(state)

    initial_step = int(state.step) + 1

    # Create the evaluation state, corresponding to a deterministic model.
    eval_net = create_model(model_name, model_hparams, deterministic=True)
    eval_state = state.replace(apply_fn=eval_net.apply)

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(num_train_steps=train_hparams.num_train_steps, writer=writer)
    profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
    hooks = [report_progress, profiler]

    # Begin training loop.
    logging.info("Starting training.")
    train_metrics = None
    for step in range(initial_step, train_hparams.num_train_steps + 1):

        # Split PRNG key, to ensure different 'randomness' for every step.
        rng, dropout_rng = jax.random.split(rng)

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            graphs = jax.tree_util.tree_map(np.asarray, next(train_iter))
            state, metrics_update = train_step(state, graphs, rngs={"dropout": dropout_rng})

            # Update metrics.
            if train_metrics is None:
                train_metrics = metrics_update
            else:
                train_metrics = train_metrics.merge(metrics_update)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)
        for hook in hooks:
            hook(step)

        # Log, if required.
        is_last_step = step == train_hparams.num_train_steps - 1
        if step % train_hparams.log_every_steps == 0 or is_last_step:
            computed_train_metrics = add_prefix_to_keys(train_metrics.compute(), "train")
            writer.write_scalars(step, computed_train_metrics)
            if wandb_logging:
                wandb.log(computed_train_metrics, step=step)
            train_metrics = None

        # Evaluate on validation and test splits, if required.
        if step % train_hparams.eval_every_steps == 0 or is_last_step:
            eval_state = eval_state.replace(params=state.params)

            splits = ["test"]
            with report_progress.timed("eval"):
                eval_metrics = evaluate_model(eval_state, datasets, splits=splits)
            for split in splits:
                computed_eval_metrics = add_prefix_to_keys(eval_metrics[split].compute(), split)
                writer.write_scalars(step, computed_eval_metrics)
                if wandb_logging:
                    wandb.log(computed_eval_metrics, step=step)

        # Checkpoint model, if required.
        # if step % train_hparams.checkpoint_every_steps == 0 or is_last_step:
        #     with report_progress.timed("checkpoint"):
        #         ckpt.save(state)
        #         if wandb_logging:
        #             artifact = wandb.Artifact(
        #                 f'{wandb.run.name}-checkpoint', type='dataset'
        #             )
        #             artifact.add_dir(checkpoint_dir)
        #             wandb.log_artifact(artifact, aliases=["latest", f"step_{step}"])

    # Finish wandb
    if wandb_logging:
        wandb.save(os.path.join(checkpoint_dir, "c*"))
        wandb.finish()

    return state
