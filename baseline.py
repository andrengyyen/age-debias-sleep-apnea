import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# TensorFlow / Keras Imports
import tensorflow as tf
import keras
from keras.utils import plot_model
from tensorflow.keras.layers import (
    Layer, Input, Conv1D, MaxPooling1D, Dropout, Dense, Flatten, 
    LSTM, MultiHeadAttention, Add, LayerNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Math / Metrics
from scipy.interpolate import splev, splrep

# --- Configuration ---
BASE_DIR = "dataset"
INTERPOLATION_RATE = 3  # Hz (Samples per second for the grid)
CONTEXT_BEFORE = 2      # Minutes
CONTEXT_AFTER = 2       # Minutes
DATA_FILE = "adversarial_data.pkl"

# --- Helpers ---

def normalize_array(arr):
    """Min-Max Normalization to scale data between 0 and 1."""
    if np.max(arr) - np.min(arr) == 0:
        return arr
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def interpolate_signal(time_points, signal_values, target_grid):
    """
    Interpolates irregular signal values onto a fixed target time grid 
    using Cubic Spline Interpolation.
    """
    return splev(target_grid, splrep(time_points, normalize_array(signal_values), k=3), ext=1)

def load_and_preprocess_data():
    """
    Loads the pickled dataset and converts irregular event data (R-peaks)
    into fixed-length time-series arrays via interpolation.
    """
    print("Loading dataset...")
    with open(os.path.join(BASE_DIR, DATA_FILE), 'rb') as f:
        data = pickle.load(f)

    total_seconds = (CONTEXT_BEFORE + 1 + CONTEXT_AFTER) * 60
    time_grid = np.arange(0, total_seconds, step=1 / float(INTERPOLATION_RATE))

    def process_split(data_list, label_list, age_list, group_list):
        processed_x = []
        for i in range(len(data_list)):
            (rri_tm, rri_signal), (ampl_tm, ampl_signal) = data_list[i]
            rri_interp = interpolate_signal(rri_tm, rri_signal, time_grid)
            ampl_interp = interpolate_signal(ampl_tm, ampl_signal, time_grid)
            processed_x.append([rri_interp, ampl_interp])
        
        return (np.array(processed_x, dtype="float32").transpose((0, 2, 1)), 
                np.array(label_list, dtype="float32"), 
                np.array(age_list, dtype="float32"), 
                np.array(group_list))

    x_train, y_train, age_train, g_train = process_split(data["o_train"], data["y_train"], data["age_train"], data["groups_train"])
    x_test, y_test, age_test, g_test = process_split(data["o_test"], data["y_test"], data["age_test"], data["groups_test"])

    return x_train, y_train, age_train, g_train, x_test, y_test, age_test, g_test

def split_train_val(x, y, age, groups, test_size=0.20):
    """
    Iterates through each unique patient and splits their data 80/20.
    Ensures the validation set has a representative sample from every patient.
    """
    print(f"\nSplitting training data {int((1-test_size)*100)}/{int(test_size*100)} per patient for validation...")
    
    x_tr, y_tr, age_tr, g_tr = [], [], [], []
    x_val, y_val, age_val, g_val = [], [], [], []

    unique_patients = np.unique(groups)
    
    for patient in unique_patients:
        idx = np.where(groups == patient)[0]
        idx_train, idx_val = train_test_split(idx, test_size=test_size, random_state=42)
        
        x_tr.append(x[idx_train])
        y_tr.append(y[idx_train])
        age_tr.append(age[idx_train])
        g_tr.append(groups[idx_train])
        
        x_val.append(x[idx_val])
        y_val.append(y[idx_val])
        age_val.append(age[idx_val])
        g_val.append(groups[idx_val])

    return (np.concatenate(x_tr), np.concatenate(y_tr), np.concatenate(age_tr), np.concatenate(g_tr),
            np.concatenate(x_val), np.concatenate(y_val), np.concatenate(age_val), np.concatenate(g_val))

# --- Model Components ---

class PositionalEncoding(Layer):
    def __init__(self, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.d_model = d_model

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.pow(10000.0, 2.0 * tf.range(self.d_model // 2, dtype=tf.float32) / self.d_model)
        angle = tf.matmul(position, div_term[tf.newaxis, :])
        sin = tf.sin(angle)
        cos = tf.cos(angle)
        pos_encoding = tf.concat([sin, cos], axis=-1)
        return pos_encoding[tf.newaxis, :, :]

def transformer_encoder_block(inputs, num_heads, key_dim, dropout_rate):
    normalized_input = LayerNormalization()(inputs)
    pos_enc_layer = PositionalEncoding(d_model=128)
    pos_enc = pos_enc_layer(normalized_input)
    transformer_input = Add()([normalized_input, pos_enc])
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(transformer_input, transformer_input)
    attention_output = Add()([transformer_input, attention_output])
    normalized_output = LayerNormalization()(attention_output)
    ff_output = Dense(128, activation='relu')(normalized_output)
    ff_output = Dense(128)(ff_output)
    encoder_output = Add()([normalized_output, ff_output])
    normalized_encoder_output = LayerNormalization()(encoder_output)
    return Dropout(dropout_rate)(normalized_encoder_output)

def create_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(inputs)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(128, kernel_size=7, strides=1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=4)(x)
    cnn_output = Dropout(0.5)(x)

    transformer_output = transformer_encoder_block(cnn_output, num_heads=2, key_dim=32, dropout_rate=0.5)

    lstm_output = LSTM(units=128, dropout=0.5, activation='tanh', return_sequences=True)(transformer_output)

    fc_output = Flatten()(lstm_output)
    fc_output = Dense(128, activation='relu')(fc_output)
    outputs = Dense(2, activation="softmax")(fc_output)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- Evaluation & Plotting ---

def evaluate_and_plot(model, history, x_test, y_test, groups_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    axes[0].plot(history["loss"], "r-", label="Training Loss", linewidth=0.5)
    axes[0].plot(history["val_loss"], "b-", label="Validation Loss", linewidth=0.5)
    axes[0].set_title("Loss over Epochs")
    axes[0].legend()

    axes[1].plot(history["accuracy"], "r-", label="Training Acc", linewidth=0.5)
    axes[1].plot(history["val_accuracy"], "b-", label="Validation Acc", linewidth=0.5)
    axes[1].set_title("Accuracy over Epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    y_score = model.predict(x_test)
    y_true_indices = np.argmax(y_test, axis=-1)
    y_pred_indices = np.argmax(y_score, axis=-1)

    C = confusion_matrix(y_true_indices, y_pred_indices, labels=[1, 0]) 
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = f1_score(y_true_indices, y_pred_indices, average='binary')
    
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    
    po = accuracy
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + TN + FP + FN) ** 2
    kappa = (po - pe) / (1 - pe)

    print(f"\nResults:\nACC: {accuracy:.4f}, Sens: {sensitivity:.4f}, Spec: {specificity:.4f}")
    print(f"F1: {f1:.4f}, AUC: {roc_auc:.4f}, Kappa: {kappa:.4f}")

    output = pd.DataFrame({
        "y_true": y_test[:, 1], 
        "y_score": y_score[:, 1], 
        "subject": groups_test
    })
    output.to_csv("model_predictions.csv", index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true_indices, y_pred_indices), annot=True, cmap='Reds', fmt='g', 
                xticklabels=['Normal', 'Apnea'], yticklabels=['Normal', 'Apnea'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    
# --- Main Execution ---

if __name__ == "__main__":
    # 1. Prepare Data
    x_train_full, y_train_full, age_train_full, g_train_full, x_test, y_test, age_test, g_test = load_and_preprocess_data()

    # 2. Split Training Data (80/20 per patient)
    x_train, y_train, age_train, g_train, x_val, y_val, age_va(
        x_train_full, y_train_full, age_train_full, g_train_full, test_size=0.20
    )
    
    print(f"Training segments: {len(y_train)} | Validation segments: {len(y_val)} | Test segments: {len(y_test)}")

    # Convert labels to One-Hot Encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_val = keras.utils.to_categorical(y_val, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    # 3. Build Model
    model = create_hybrid_model(input_shape=x_train.shape[1:])
    model.summary()

    # 4. Compile Model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # 5. Define Callbacks
    os.makedirs('model', exist_ok=True)
    W_FILE = f'model/baseline.model.keras'
    callbacks_list = [
        ModelCheckpoint(filepath=W_FILE, monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1),
        CSVLogger('training_log.csv', separator=',', append=True)
    ]

    # 6. Train (Using isolated validation split)
    history = model.fit(
        x_train, y_train, 
        batch_size=128, 
        epochs=100, 
        validation_data=(x_val, y_val), 
        callbacks=callbacks_list
    )

    # 7. Evaluate (Loading best weights prior to evaluation to avoid overfitting)
    print("\nLoading best weights for final evaluation on test set...")
    model.load_weights(W_FILE)
    evaluate_and_plot(model, history.history, x_test, y_test, g_test)