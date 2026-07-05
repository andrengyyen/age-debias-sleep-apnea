"""
FSCL ablation study for age-bias mitigation in OSA detection.

Builds on the CNN-Transformer-LSTM baseline (Pham et al. 2025) and adds the
Fair Supervised Contrastive Loss of Park et al. (CVPR 2022).

This script contains:
  1. A corrected FSCL / FSCL+ loss (positives kept in the denominator).
  2. A parameterised model builder (optional separate projection head).
  3. A custom training model wrapping CE + lambda * FSCL.
  4. Fairness metrics (per age group + Equalized-Odds-style gaps).
  5. A grid-search ablation loop over (lambda, temperature, group_norm,
     projection head, batch size, temperature ...) across multiple seeds,
     writing a results CSV and printing an accuracy-vs-fairness summary.

DATA:  re-uses existing pipeline in baseline.py.
"""

import os
import gc
import json
import itertools
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from tensorflow.keras.layers import (
    Layer, Input, Conv1D, MaxPooling1D, Dropout, Dense,
    LSTM, MultiHeadAttention, Add, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# --- Reuse data pipeline -------------------------------------------------
from baseline import load_and_preprocess_data, split_train_val
SEED = 0
WEIGHTS_OUT = "model/best_fscl_final.weights.h5"
CHOSEN_FSCL = dict(lambda_fscl=1.0, temperature=0.1, group_norm=True,
                   use_projection_head=False, batch_size=128,
                   embed_dim=64, proj_dim=64, lr=1e-3, dropout=0.5, n_age_groups=2)
EPOCHS = 80
PATIENCE = 15
# =============================================================================
# 1. Corrected FSCL / FSCL+ loss
# =============================================================================

def fair_supcon_loss(z, y, a, temperature=0.1, n_classes=2, n_age_groups=2,
                     group_norm=True):
    """Fair Supervised Contrastive Loss (Park et al., CVPR 2022).

    Positives  : same diagnostic class (any age)            -> pulled together
    Fair negs  : different class AND same age group         -> pushed apart
    TSG pairs  : different class AND different age group     -> NOT in the loss

    The denominator follows SupCon Eq.2 with the negative set restricted to
    same-age, i.e. it is {positives} U {fair negatives}. Crucially the
    positive terms remain inside the denominator.

    group_norm=True -> FSCL+ (group-wise normalisation by group cardinality).
    """
    z = tf.math.l2_normalize(z, axis=1)
    B = tf.shape(z)[0]
    logits = tf.matmul(z, z, transpose_b=True) / temperature

    y_i, y_j = y[:, None], y[None, :]
    a_i, a_j = a[:, None], a[None, :]

    not_self = 1.0 - tf.eye(B)
    same_y = tf.cast(tf.equal(y_i, y_j), tf.float32)
    same_a = tf.cast(tf.equal(a_i, a_j), tf.float32)

    pos_mask = same_y * not_self                  # IG + SG positives
    fair_neg = (1.0 - same_y) * same_a            # TG negatives (same age, diff class)
    denom_mask = pos_mask + fair_neg              # excludes TSG (diff age, diff class)

    # numerical stability
    logits = logits - tf.stop_gradient(tf.reduce_max(logits, axis=1, keepdims=True))
    exp_logits = tf.exp(logits)

    # FIX: positives are part of the denominator (proper log-softmax)
    denom = tf.reduce_sum(exp_logits * denom_mask, axis=1, keepdims=True) + 1e-12
    log_prob = logits - tf.math.log(denom)

    pos_per_anchor = tf.reduce_sum(pos_mask, axis=1)
    has_pos = tf.cast(pos_per_anchor > 0.0, tf.float32)
    mean_log_prob_pos = tf.reduce_sum(pos_mask * log_prob, axis=1) / (pos_per_anchor + 1e-12)

    per_anchor = -mean_log_prob_pos * has_pos     # drop anchors with no positives

    if group_norm:  # FSCL+
        gid = tf.cast(y, tf.int32) * n_age_groups + tf.cast(a, tf.int32)
        counts = tf.cast(tf.math.bincount(gid, minlength=n_classes * n_age_groups), tf.float32)
        w = 1.0 / (tf.gather(counts, gid) + 1e-12)
        return tf.reduce_sum(per_anchor * w) / (tf.reduce_sum(w * has_pos) + 1e-12)

    return tf.reduce_sum(per_anchor) / (tf.reduce_sum(has_pos) + 1e-12)


# =============================================================================
# 2. Architecture (CNN-Transformer-LSTM) with optional projection head
# =============================================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(10000.0, 2.0 * tf.range(self.d_model // 2, dtype=tf.float32) / self.d_model)
        angle = tf.matmul(position, div_term[tf.newaxis, :])
        pos = tf.concat([tf.sin(angle), tf.cos(angle)], axis=-1)
        return pos[tf.newaxis, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


def transformer_encoder_block(x, num_heads=2, key_dim=32, dropout_rate=0.5):
    norm1 = LayerNormalization()(x)
    pos = PositionalEncoding(d_model=128)(norm1)
    t_in = Add()([norm1, pos])
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(t_in, t_in)
    attn = Add()([t_in, attn])
    norm2 = LayerNormalization()(attn)
    ff = Dense(128, activation="relu")(norm2)
    ff = Dense(128)(ff)
    out = Add()([norm2, ff])
    out = LayerNormalization()(out)
    return Dropout(dropout_rate)(out)


def build_model(input_shape, embed_dim=64, proj_dim=64,
                use_projection_head=True, dropout=0.5):
    """Returns Model(inputs) -> [logits, z].

    use_projection_head=True  : classifier reads the encoder representation `h`;
                                a separate *linear* head g(h)=z feeds the FSCL
                                loss (faithful to Park et al.). Recommended.
    use_projection_head=False : your original setup -- the ReLU 'embedding' is
                                shared by both the classifier and FSCL.
    """
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 7, padding="same", activation="relu")(inputs)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 7, padding="same", activation="relu")(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(dropout)(x)
    x = transformer_encoder_block(x, dropout_rate=dropout)
    x = LSTM(128, return_sequences=False, dropout=dropout)(x)

    if use_projection_head:
        h = Dense(embed_dim, activation="relu", name="representation")(x)
        z = Dense(proj_dim, activation=None, name="projection")(h)   # linear head for FSCL
        logits = Dense(2, activation="softmax", name="apnea_output")(h)
    else:
        h = Dense(embed_dim, activation="relu", name="embedding")(x)
        z = h                                                        # shared (original)
        logits = Dense(2, activation="softmax", name="apnea_output")(h)

    return Model(inputs=inputs, outputs=[logits, z])


class FSCLApneaModel(tf.keras.Model):
    def __init__(self, base_model, lambda_fscl=0.1, temperature=0.1,
                 n_classes=2, n_age_groups=2, group_norm=True, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.lambda_fscl = lambda_fscl
        self.temp = temperature
        self.n_classes = n_classes
        self.n_age_groups = n_age_groups
        self.group_norm = group_norm
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.ce_tracker = tf.keras.metrics.Mean(name="ce")
        self.fscl_tracker = tf.keras.metrics.Mean(name="fscl")
        self.acc_tracker = tf.keras.metrics.CategoricalAccuracy(name="apnea_acc")

    def call(self, x, training=False):
        logits, _ = self.base_model(x, training=training)
        return logits

    def _compute(self, x, y_dict, training):
        y_apnea = y_dict["apnea_output"]
        y_age = y_dict["age_group"]
        y_lab = tf.argmax(y_apnea, axis=-1)
        a_lab = tf.argmax(y_age, axis=-1)
        logits, z = self.base_model(x, training=training)
        ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_apnea, logits))
        fscl = fair_supcon_loss(z, y_lab, a_lab, temperature=self.temp,
                                n_classes=self.n_classes, n_age_groups=self.n_age_groups,
                                group_norm=self.group_norm)
        total = ce + self.lambda_fscl * fscl
        return total, ce, fscl, logits, y_apnea

    def train_step(self, data):
        x, y_dict = data
        with tf.GradientTape() as tape:
            total, ce, fscl, logits, y_apnea = self._compute(x, y_dict, training=True)
        grads = tape.gradient(total, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.base_model.trainable_variables))
        self.loss_tracker.update_state(total)
        self.ce_tracker.update_state(ce)
        self.fscl_tracker.update_state(fscl)
        self.acc_tracker.update_state(y_apnea, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y_dict = data
        total, ce, fscl, logits, y_apnea = self._compute(x, y_dict, training=False)
        self.loss_tracker.update_state(total)
        self.ce_tracker.update_state(ce)
        self.fscl_tracker.update_state(fscl)
        self.acc_tracker.update_state(y_apnea, logits)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.ce_tracker, self.fscl_tracker, self.acc_tracker]


# =============================================================================
# 3. Fairness metrics
# =============================================================================

def _group_metrics(t, p, pr):
    cm = confusion_matrix(t, p, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel()
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    sens = tp / (tp + fn) if (tp + fn) else 0.0          # TPR / recall
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = f1_score(t, p, zero_division=0)
    try:
        fpr_c, tpr_c, _ = roc_curve(t, pr)
        auc_v = auc(fpr_c, tpr_c)
    except Exception:
        auc_v = float("nan")
    return dict(acc=acc, sens=sens, spec=spec, fpr=fpr, f1=f1, auc=auc_v)


def fairness_report(model, x, y_onehot, age, age_names={0: "young", 1: "old"}):
    """Returns a flat dict of overall + per-group metrics and fairness gaps."""
    probs = model.predict(x, verbose=0)
    y_true = np.argmax(y_onehot, axis=-1)
    y_pred = np.argmax(probs, axis=-1)
    age = np.asarray(age).astype(int)

    out = {}
    overall = _group_metrics(y_true, y_pred, probs[:, 1])
    for k, v in overall.items():
        out[f"overall_{k}"] = v

    per = {}
    for av, name in age_names.items():
        m = age == av
        if np.any(m):
            per[name] = _group_metrics(y_true[m], y_pred[m], probs[m, 1])
            for k, v in per[name].items():
                out[f"{name}_{k}"] = v

    # Fairness gaps between the two groups (smaller = fairer)
    if len(per) == 2:
        g0, g1 = list(per.values())
        tpr_gap = abs(g0["sens"] - g1["sens"])          # equal opportunity gap
        fpr_gap = abs(g0["fpr"] - g1["fpr"])
        out["eo_gap"] = max(tpr_gap, fpr_gap)           # equalized-odds gap
        out["eo_gap_mean"] = 0.5 * (tpr_gap + fpr_gap)
        out["tpr_gap"] = tpr_gap
        out["fpr_gap"] = fpr_gap
        out["f1_gap"] = abs(g0["f1"] - g1["f1"])
        out["acc_gap"] = abs(g0["acc"] - g1["acc"])
        out["auc_gap"] = abs(g0["auc"] - g1["auc"])
    return out


# =============================================================================
# 4. Train + evaluate a single configuration
# =============================================================================

def run_one(cfg, seed, data, epochs, patience, workdir="ablation_ckpts"):
    (x_tr, y_tr_d, x_val, y_val_d, x_te, y_te_oh, a_val, a_te) = data
    tf.keras.utils.set_random_seed(seed)
    os.makedirs(workdir, exist_ok=True)
    wfile = os.path.join(workdir, f"tmp_{cfg['tag']}_s{seed}.weights.h5")

    base = build_model(
        input_shape=x_tr.shape[1:],
        embed_dim=cfg.get("embed_dim", 64),
        proj_dim=cfg.get("proj_dim", 64),
        use_projection_head=cfg.get("use_projection_head", True),
        dropout=cfg.get("dropout", 0.5),
    )
    model = FSCLApneaModel(
        base_model=base,
        lambda_fscl=cfg["lambda_fscl"],
        temperature=cfg["temperature"],
        group_norm=cfg.get("group_norm", True),
        n_age_groups=cfg.get("n_age_groups", 2),
    )
    model.compile(optimizer=keras.optimizers.Adam(cfg.get("lr", 1e-3)))
    model(x_tr[:1])  # build

    cbs = [
        ModelCheckpoint(wfile, monitor="val_loss", save_best_only=True,
                        save_weights_only=True, mode="min", verbose=0),
        EarlyStopping(monitor="val_loss", patience=patience, mode="min",
                      restore_best_weights=True, verbose=0),
    ]
    model.fit(x_tr, y_tr_d, validation_data=(x_val, y_val_d),
              batch_size=cfg.get("batch_size", 128), epochs=epochs,
              callbacks=cbs, verbose=0)
    if os.path.exists(wfile):
        model.load_weights(wfile)

    val_m = fairness_report(model, x_val, y_val_d["apnea_output"], a_val)
    test_m = fairness_report(model, x_te, y_te_oh, a_te)

    row = {f"val_{k}": v for k, v in val_m.items()}
    row.update({f"test_{k}": v for k, v in test_m.items()})
    # validation-only selection score: reward accuracy, penalise unfairness.
    row["val_selection_score"] = val_m["overall_acc"] - cfg.get("beta", 1.0) * val_m.get("eo_gap", 1.0)

    # cleanup to keep memory flat across the grid
    try:
        os.remove(wfile)
    except OSError:
        pass
    del model, base
    keras.backend.clear_session()
    gc.collect()
    return row


# =============================================================================
# 5. Ablation grid
# =============================================================================

# Each key may hold a list; the Cartesian product is swept. Put a single value
# in a list to hold it fixed. lambda_fscl=0.0 == the CE-only baseline.
ABLATION_GRID = {
    "lambda_fscl":          [0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
    "temperature":          [0.05, 0.07, 0.1, 0.2], # 0.1 will be chosen by following contrastive learning studies
    "group_norm":           [True, False],          # FSCL+ vs FSCL
    "use_projection_head":  [True, False],          # separate head vs shared ReLU
    # held fixed (single-element lists):
    "batch_size":           [128],
    "embed_dim":            [64],
    "proj_dim":             [64],
    "lr":                   [1e-3],
    "dropout":              [0.5],
    "beta":                 [1.0],                  # selection trade-off weight
    "n_age_groups":         [2],
}

SEEDS = [0, 1, 2]      # average over seeds; contrastive losses are seed-sensitive
EPOCHS = 100
PATIENCE = 15
RESULTS_CSV = "fscl_ablation_results.csv"


def expand_grid(grid):
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        cfg["tag"] = "_".join(
            f"{k}{cfg[k]}" for k in ["lambda_fscl", "temperature", "group_norm",
                                     "use_projection_head", "batch_size"]
        )
        yield cfg

def main():
    tf.keras.utils.set_random_seed(SEED)
 
    # --- data (same patient-level split used in the ablation) ---------------
    x_tr_full, y_tr_full, a_tr_full, g_tr_full, x_te, y_te, a_te, g_te = load_and_preprocess_data()
    x_tr, y_tr, a_tr, g_tr, x_val, y_val, a_val, g_val = split_train_val(
        x_tr_full, y_tr_full, a_tr_full, g_tr_full, test_size=0.20)
 
    y_tr_d = {"apnea_output": keras.utils.to_categorical(y_tr, 2),
              "age_group":    keras.utils.to_categorical(a_tr, 2)}
    y_val_d = {"apnea_output": keras.utils.to_categorical(y_val, 2),
               "age_group":    keras.utils.to_categorical(a_val, 2)}
    y_te_oh = keras.utils.to_categorical(y_te, 2)
    a_te = np.asarray(a_te).astype(int)
 
    print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)} segments")
 
    # --- model --------------------------------------------------------------
    base = build_model(input_shape=x_tr.shape[1:],
                       embed_dim=CHOSEN_FSCL["embed_dim"],
                       proj_dim=CHOSEN_FSCL["proj_dim"],
                       use_projection_head=CHOSEN_FSCL["use_projection_head"],
                       dropout=CHOSEN_FSCL["dropout"])
    model = FSCLApneaModel(base_model=base,
                           lambda_fscl=CHOSEN_FSCL["lambda_fscl"],
                           temperature=CHOSEN_FSCL["temperature"],
                           group_norm=CHOSEN_FSCL["group_norm"],
                           n_age_groups=CHOSEN_FSCL["n_age_groups"])
    model.compile(optimizer=keras.optimizers.Adam(CHOSEN_FSCL["lr"]))
    model(x_tr[:1])  # build variables
 
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    cbs = [ModelCheckpoint(WEIGHTS_OUT, monitor="val_loss", save_best_only=True,
                           save_weights_only=True, mode="min", verbose=1),
           EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min",
                         restore_best_weights=True, verbose=1)]
 
    print("Training chosen FSCL config...")
    model.fit(x_tr, y_tr_d, validation_data=(x_val, y_val_d),
              batch_size=CHOSEN_FSCL["batch_size"], epochs=EPOCHS,
              callbacks=cbs, verbose=2)
    model.load_weights(WEIGHTS_OUT)
    print(f"\nSaved weights to {WEIGHTS_OUT}")
 
    # --- stratified evaluation ---------------------------------------------
    m = fairness_report(model, x_te, y_te_oh, a_te)
    rows = [
        ["Overall", m["overall_acc"], m["overall_sens"], m["overall_spec"],
         m["overall_f1"], m["overall_auc"]],
        ["Young",   m.get("young_acc"), m.get("young_sens"), m.get("young_spec"),
         m.get("young_f1"), m.get("young_auc")],
        ["Old",     m.get("old_acc"), m.get("old_sens"), m.get("old_spec"),
         m.get("old_f1"), m.get("old_auc")],
    ]
    df = pd.DataFrame(rows, columns=["Group", "Acc", "Sens", "Spec", "F1", "AUC"])
    print("\n" + "=" * 70 + "\nFSCL TEST EVALUATION (chosen config)\n" + "=" * 70)
    print(df.to_string(index=False,
                       formatters={c: "{:.2%}".format for c in df.columns[1:]}))
    print(f"\nEqualized-odds gap : {m['eo_gap']:.4f}  "
          f"(TPR gap {m['tpr_gap']:.4f}, FPR gap {m['fpr_gap']:.4f})")
    print(f"F1 gap             : {m['f1_gap']:.4f}")
    print(f"Accuracy gap       : {m['acc_gap']:.4f}")

if __name__ == "__main__":
    main()
