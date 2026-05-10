from pathlib import Path

import numpy as np
import tensorflow as tf

from .metrics import summarize_metrics


def _extract_next_step_targets(logits, batch):
    next_logits = logits[:, :-1]
    next_targets = tf.cast(batch["responses"][:, 1:], tf.float32)
    next_mask = tf.cast(batch["mask"][:, 1:], tf.float32)
    valid_logits = tf.boolean_mask(next_logits, next_mask > 0)
    valid_targets = tf.boolean_mask(next_targets, next_mask > 0)
    return valid_logits, valid_targets


def train_one_epoch(model, optimizer, dataset):
    all_targets = []
    all_predictions = []
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for batch in dataset:
        with tf.GradientTape() as tape:
            logits = model(batch, training=True)
            valid_logits, valid_targets = _extract_next_step_targets(logits, batch)
            batch_loss = loss_fn(valid_targets, valid_logits)
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        all_targets.append(valid_targets.numpy())
        all_predictions.append(tf.sigmoid(valid_logits).numpy())

    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    return summarize_metrics(targets, predictions)


def evaluate(model, dataset):
    all_targets = []
    all_predictions = []

    for batch in dataset:
        logits = model(batch, training=False)
        valid_logits, valid_targets = _extract_next_step_targets(logits, batch)
        all_targets.append(valid_targets.numpy())
        all_predictions.append(tf.sigmoid(valid_logits).numpy())

    targets = np.concatenate(all_targets, axis=0)
    predictions = np.concatenate(all_predictions, axis=0)
    return summarize_metrics(targets, predictions)


def fit_and_evaluate(model, train_bundle, valid_bundle, test_bundle, config):
    train_dataset = train_bundle.to_tf_dataset(
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    valid_dataset = valid_bundle.to_tf_dataset(
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    test_dataset = test_bundle.to_tf_dataset(
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.training.learning_rate)
    checkpoint_dir = config.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=1)

    best_valid_auc = -np.inf
    best_epoch = -1
    patience_count = 0
    history = []

    for epoch_idx in range(config.training.epochs):
        train_metrics = train_one_epoch(model, optimizer, train_dataset)
        valid_metrics = evaluate(model, valid_dataset)

        history.append(
            {
                "epoch": epoch_idx + 1,
                "train": train_metrics,
                "valid": valid_metrics,
            }
        )

        if valid_metrics["auc"] > best_valid_auc:
            best_valid_auc = valid_metrics["auc"]
            best_epoch = epoch_idx + 1
            patience_count = 0
            manager.save()
        else:
            patience_count += 1
            if patience_count >= config.training.patience:
                break

    latest_checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()

    test_metrics = evaluate(model, test_dataset)

    return {
        "task_name": config.task_name,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_valid_auc": float(best_valid_auc),
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": latest_checkpoint,
    }
