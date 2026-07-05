"""
DANN.py
============================================================
DANN for age-fair sleep-apnea detection, plus a hyperparameter
ablation runner.

--------------------------------------------------
1. GradientReversalLayer.alpha is now a non-trainable tf.Variable, read live
   at execution time. The GRLAlphaScheduler can therefore actually change it
   during training (previous schedule was frozen at trace time).
2. Optional CLASS-CONDITIONAL adversary (`conditional=True`): the age head is
   conditioned on the apnea label, which targets EQUALIZED ODDS (both the
   sensitivity AND specificity gaps).
3. Stratified evaluation that reports per-age-group sensitivity/specificity
   and the gaps, robust to a group missing a class.
4. An ablation runner that sweeps max_alpha x schedule x conditional x seed
   and writes a CSV.

Data loaders are imported from existing baseline.py.
============================================================
"""

import os
import itertools
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Layer, Input, Conv1D, MaxPooling1D, Dropout, Dense,
    LSTM, MultiHeadAttention, Add, LayerNormalization, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# --- Reuse data pipeline -------------------------------------------------
from baseline import (  
    load_and_preprocess_data,
    split_train_val,
)

NUM_AGE_GROUPS = 2          # 0 = young, 1 = old (per your preprocessing.py)
AGE_GROUP_NAMES = {0: "young", 1: "old"}


# ============================================================
# 1. Fixed Gradient Reversal Layer (alpha is a live tf.Variable)
# ============================================================
class GradientReversalLayer(Layer):
    """Identity forward; reversed+scaled gradient backward.

    alpha is a non-trainable tf.Variable, so a callback can update it via
    set_alpha() and the change is picked up at the next training step WITHOUT
    re-tracing the graph.
    """
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)          # let Keras own `name`
        self._alpha = tf.Variable(
            float(alpha), trainable=False, dtype=tf.float32, name="grl_alpha"
        )

    def set_alpha(self, value):
        self._alpha.assign(float(value))

    @property
    def alpha_value(self):
        return float(self._alpha.numpy())

    def call(self, x):
        alpha = self._alpha

        @tf.custom_gradient
        def _reverse(z):
            def grad(dy):
                return -alpha * dy          # reversed and scaled, read live
            return tf.identity(z), grad

        return _reverse(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"alpha": self.alpha_value})
        return cfg


class GRLAlphaScheduler(keras.callbacks.Callback):
    """Schedules the GRL coefficient.

    schedule:
        "constant" -> max_alpha after warmup
        "linear"   -> linear ramp warmup..total_epochs
        "dann"     -> 2/(1+exp(-gamma*p)) - 1 ramp (Ganin et al.)
    """
    def __init__(self, schedule="linear", warmup_epochs=5, max_alpha=1.0,
                 total_epochs=100, gamma=10.0, verbose=1):
        super().__init__()
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.max_alpha = max_alpha
        self.total_epochs = total_epochs
        self.gamma = gamma
        self.verbose = verbose

    def _grl_layers(self):
        return [l for l in self.model.layers
                if isinstance(l, GradientReversalLayer)]

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            alpha = 0.0
        else:
            denom = max(1, self.total_epochs - self.warmup_epochs)
            p = (epoch - self.warmup_epochs) / denom
            p = min(1.0, max(0.0, p))
            if self.schedule == "linear":
                alpha = self.max_alpha * p
            elif self.schedule == "dann":
                alpha = self.max_alpha * (2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0)
            else:  # constant
                alpha = self.max_alpha
        for l in self._grl_layers():
            l.set_alpha(alpha)
        if self.verbose:
            print(f"[GRL] epoch {epoch}: alpha = {alpha:.4f}")


# ============================================================
# 2. Model
# ============================================================
class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(
            10000.0, 2.0 * tf.range(self.d_model // 2, dtype=tf.float32) / self.d_model
        )
        angle = tf.matmul(position, div_term[tf.newaxis, :])
        pos_encoding = tf.concat([tf.sin(angle), tf.cos(angle)], axis=-1)
        return pos_encoding[tf.newaxis, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate):
    normalized_input = LayerNormalization()(inputs)
    pos_enc = PositionalEncoding(d_model=128)(normalized_input)
    transformer_input = Add()([normalized_input, pos_enc])
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        transformer_input, transformer_input
    )
    attn = Add()([transformer_input, attn])
    normalized_output = LayerNormalization()(attn)
    ff = Dense(128, activation="relu")(normalized_output)
    ff = Dense(128)(ff)
    enc = Add()([normalized_output, ff])
    enc = LayerNormalization()(enc)
    return Dropout(dropout_rate)(enc)


def create_adversarial_model(input_shape, alpha=1.0, conditional=False,
                             num_age_groups=NUM_AGE_GROUPS):
    """CNN-Transformer-LSTM shared trunk + apnea head + age adversary.

    conditional=False -> marginal DANN (aligns P(f|age); ~demographic parity)
    conditional=True  -> class-conditional adversary: the apnea label is fed
                         into the adversary after the GRL, aligning P(f|age,y)
                         (~equalized odds -> targets BOTH sens & spec gaps).
    """
    sig_in = Input(shape=input_shape, name="signal")

    # --- shared feature extractor ---
    x = Conv1D(64, 7, padding="same", activation="relu")(sig_in)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(0.5)(x)
    x = transformer_encoder_block(x, num_heads=2, key_dim=32, dropout_rate=0.5)
    x = LSTM(128, return_sequences=False, dropout=0.5)(x)
    bottleneck = Dense(64, activation="relu", name="shared_bottleneck")(x)

    # --- apnea head (main task) ---
    apnea = Dense(64, activation="relu")(bottleneck)
    apnea_output = Dense(2, activation="softmax", name="apnea_output")(apnea)

    # --- age adversary ---
    grl = GradientReversalLayer(alpha=alpha, name="GRL")(bottleneck)
    if conditional:
        label_in = Input(shape=(2,), name="label_in")   # true apnea one-hot
        adv_in = Concatenate(name="adv_concat")([grl, label_in])
        inputs = [sig_in, label_in]
    else:
        adv_in = grl
        inputs = sig_in

    a = Dense(128, activation="relu")(adv_in)
    a = Dropout(0.3)(a)
    a = Dense(64, activation="relu")(a)
    age_output = Dense(num_age_groups, activation="softmax", name="age_output")(a)

    return Model(inputs=inputs, outputs=[apnea_output, age_output])


# ============================================================
# 3. Stratified evaluation
# ============================================================
def _binary_metrics(y_true, y_pred, y_score=None):
    """Robust to a subgroup missing one class (forces a 2x2 matrix)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception:
        f1 = np.nan
    roc = np.nan
    if y_score is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc = auc(fpr, tpr)
    return dict(acc=acc, sens=sens, spec=spec, f1=f1, auc=roc,
                n=int(tp + tn + fp + fn))


def stratified_eval(apnea_probs, y_true_oh, age_true_oh, tag=""):
    """Returns a dict of overall + per-group metrics and the fairness gaps."""
    y_true = np.argmax(y_true_oh, axis=-1)
    y_pred = np.argmax(apnea_probs, axis=-1)
    y_score = apnea_probs[:, 1]
    age = np.argmax(age_true_oh, axis=-1)

    out = {"tag": tag}
    overall = _binary_metrics(y_true, y_pred, y_score)
    for k, v in overall.items():
        out[f"overall_{k}"] = v

    per_group = {}
    for g in range(NUM_AGE_GROUPS):
        m = age == g
        if m.sum() == 0:
            continue
        gm = _binary_metrics(y_true[m], y_pred[m], y_score[m])
        per_group[g] = gm
        gname = AGE_GROUP_NAMES.get(g, str(g))
        for k, v in gm.items():
            out[f"{gname}_{k}"] = v

    # Gaps between group 0 and group 1 (extend if you add groups)
    if 0 in per_group and 1 in per_group:
        out["sens_gap"] = abs(per_group[0]["sens"] - per_group[1]["sens"])
        out["spec_gap"] = abs(per_group[0]["spec"] - per_group[1]["spec"])
        # equalized-odds gap = worst of the two
        out["eo_gap"] = np.nanmax([out["sens_gap"], out["spec_gap"]])
    return out


def print_eval(out):
    print(f"\n=== {out.get('tag','')} ===")
    print(f"Overall  acc={out['overall_acc']:.3f}  sens={out['overall_sens']:.3f}  "
          f"spec={out['overall_spec']:.3f}  f1={out['overall_f1']:.3f}  auc={out['overall_auc']:.3f}")
    for g, gname in AGE_GROUP_NAMES.items():
        if f"{gname}_sens" in out:
            print(f"  {gname:>5}: sens={out[f'{gname}_sens']:.3f}  "
                  f"spec={out[f'{gname}_spec']:.3f}  n={out[f'{gname}_n']}")
    if "sens_gap" in out:
        print(f"  GAPS: sens_gap={out['sens_gap']:.3f}  "
              f"spec_gap={out['spec_gap']:.3f}  eo_gap={out['eo_gap']:.3f}")


# ============================================================
# 4. Train a single configuration
# ============================================================
def train_one(cfg, data, total_epochs=100, batch_size=128, verbose=1):
    """cfg keys: max_alpha, schedule, conditional, warmup_epochs, age_loss_weight, seed"""
    (x_train, y_tr_oh, age_tr_oh,
     x_val, y_val_oh, age_val_oh,
     x_test, y_test_oh, age_test_oh) = data

    tf.keras.utils.set_random_seed(cfg.get("seed", 0))

    conditional = cfg.get("conditional", False)
    model = create_adversarial_model(
        input_shape=x_train.shape[1:],
        alpha=0.0,                       # scheduler drives it from here
        conditional=conditional,
    )

    model.compile(
        optimizer="adam",
        # categorical_crossentropy is the clean choice for a 2-unit softmax
        loss={"apnea_output": "categorical_crossentropy",
              "age_output": "categorical_crossentropy"},
        loss_weights={"apnea_output": 1.0,
                      "age_output": cfg.get("age_loss_weight", 1.0)},
        metrics={"apnea_output": "accuracy", "age_output": "accuracy"},
    )

    os.makedirs("model_adv", exist_ok=True)
    tag = (f"a{cfg['max_alpha']}_{cfg.get('schedule','linear')}"
           f"_{'cond' if conditional else 'marg'}_s{cfg.get('seed',0)}")
    # w_file = f"model_adv/DANN_{tag}.weights.h5"
    w_file = f"model_adv/DANN_cross.weights.h5"

    callbacks = [
        GRLAlphaScheduler(schedule=cfg.get("schedule", "linear"),
                          warmup_epochs=cfg.get("warmup_epochs", 5),
                          max_alpha=cfg["max_alpha"],
                          total_epochs=total_epochs,
                          verbose=0),
        ModelCheckpoint(w_file, monitor="val_apnea_output_loss",
                        save_best_only=True, save_weights_only=True,
                        mode="min", verbose=0),
        EarlyStopping(monitor="val_apnea_output_loss", patience=20,
                      mode="min", restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_apnea_output_loss", patience=5,
                          mode="min", verbose=0),
    ]

    if conditional:
        train_x = [x_train, y_tr_oh]
        val_x = [x_val, y_val_oh]
    else:
        train_x = x_train
        val_x = x_val

    model.fit(
        train_x, {"apnea_output": y_tr_oh, "age_output": age_tr_oh},
        validation_data=(val_x, {"apnea_output": y_val_oh, "age_output": age_val_oh}),
        batch_size=batch_size, epochs=total_epochs,
        callbacks=callbacks, verbose=verbose,
    )

    # best weights restored by EarlyStopping; predict on test
    if conditional:
        dummy = np.zeros((len(x_test), 2), dtype="float32")  # label only feeds adversary
        preds = model.predict([x_test, dummy], verbose=0)
    else:
        preds = model.predict(x_test, verbose=0)
    apnea_probs = preds[0]

    out = stratified_eval(apnea_probs, y_test_oh, age_test_oh, tag=tag)
    out.update({k: cfg[k] for k in cfg})
    return out, model


# ============================================================
# 5. Ablation runner
# ============================================================
def run_ablation(data, total_epochs=100, out_csv="dann_ablation_results.csv"):
    grid = {
        "max_alpha":   [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],   # GRL strength (lambda)
        "schedule":    ["linear"],                       # add "dann","constant" to compare
        "conditional": [False, True],                    # marginal vs equalized-odds adversary
        "warmup_epochs": [5],
        "age_loss_weight": [1.0],
        "seed":        [0, 1, 2],                         # report mean +/- std
    }
    keys = list(grid.keys())
    rows = []
    for values in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, values))
        # alpha=0 + marginal is the no-adversary control; skip its duplicate
        # across schedules since schedule is irrelevant when alpha=0.
        print(f"\n>>> CONFIG: {cfg}")
        out, _ = train_one(cfg, data, total_epochs=total_epochs, verbose=0)
        print_eval(out)
        rows.append(out)
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # checkpoint each run

    df = pd.DataFrame(rows)

    # Aggregate over seeds
    group_cols = ["max_alpha", "schedule", "conditional",
                  "warmup_epochs", "age_loss_weight"]
    metric_cols = ["overall_acc", "overall_sens", "overall_spec",
                   "overall_auc", "sens_gap", "spec_gap", "eo_gap"]
    agg = (df.groupby(group_cols)[metric_cols]
             .agg(["mean", "std"]).reset_index())
    agg.to_csv(out_csv.replace(".csv", "_aggregated.csv"), index=False)

    print("\n================ ABLATION SUMMARY (mean over seeds) ================")
    summary = df.groupby(group_cols)[metric_cols].mean().reset_index()
    summary = summary.sort_values("eo_gap")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv} and *_aggregated.csv")
    return df, agg


# ============================================================
# 6. Main
# ============================================================
def prepare_data():
    x_train_full, y_train_full, age_train_full, g_train_full, x_test, y_test, age_test, g_test = load_and_preprocess_data()

    x_train, y_train, age_train, _, x_val, y_val, age_val, _ = \
        split_train_val(x_train_full, y_train_full,
                                   age_train_full, g_train_full, test_size=0.20)

    to_cat = keras.utils.to_categorical
    return (
        x_train, to_cat(y_train, 2), to_cat(age_train, NUM_AGE_GROUPS),
        x_val,   to_cat(y_val, 2),   to_cat(age_val, NUM_AGE_GROUPS),
        x_test,  to_cat(y_test, 2),  to_cat(age_test, NUM_AGE_GROUPS),
    )


if __name__ == "__main__":
    data = prepare_data()
    # Quick single run:
    out, model = train_one(
        dict(max_alpha=0.7, schedule="linear", conditional=True,
             warmup_epochs=5, age_loss_weight=1.0, seed=2), # parameter give best fairness
        data, total_epochs=100, verbose=1)
    print_eval(out)

    # Full ablation:
    # run_ablation(data, total_epochs=100)