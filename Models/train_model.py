import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
from pathlib import Path
from tensorflow.keras.optimizers import Adam

# Load before training
model_before = load_model('Model/keras_model.h5')

# Compile model
model_before.compile(
    optimizer= Adam(learning_rate=3e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Preprocessing with Data Augmentation for training
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    channel_shift_range=30.0,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Test set: only rescaling (no augmentation)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'Dataset/train',
    target_size=(200, 200),
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)
test_data = test_gen.flow_from_directory(
    'Dataset/test',
    target_size=(200, 200),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

# Train model
ASL_history = model_before.fit(
    train_data,
    validation_data=test_data,
    epochs=30,   # You can go higher, early stopping will stop when needed
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# Save updated model
model_before.save('Model/keras_model_updated.h5')

# Load after training
model_after = load_model('Model/keras_model_updated.h5')
print("\n MODEL SUMMARY ")
model_after.summary()

# Save final metrics
metrics = ASL_history.history
final_metrics = {key: float(values[-1]) for key, values in metrics.items()}
final_metrics["model_name"] = "trinh nhat huy's model"

results_path = Path("Results") / "tnh_model.json"
results_path.parent.mkdir(exist_ok=True)
with open(results_path, "w") as f:
    json.dump(final_metrics, f, indent=4)

print(f"Saved results to {results_path}")
