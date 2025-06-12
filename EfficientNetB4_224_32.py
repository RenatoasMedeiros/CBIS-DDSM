# %%
# %%
# Import necessary libraries at the top
import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from datetime import datetime

# --- Configuration & Constants ---
MODEL_NAME = 'EfficientNetB4_Binary'

BASE_DATASET_PATH = './k_CBIS-DDSM/'
CALC_METADATA_CSV_PATH = os.path.join(BASE_DATASET_PATH, 'calc_case(with_jpg_img).csv')
MASS_METADATA_CSV_PATH = os.path.join(BASE_DATASET_PATH, 'mass_case(with_jpg_img).csv')

IMAGE_ROOT_DIR = BASE_DATASET_PATH
ACTUAL_IMAGE_FILES_BASE_DIR = os.path.join(IMAGE_ROOT_DIR, 'jpg_img')

# Column in CSV that conceptually should point to ROIs, even if paths are flawed
CONCEPTUAL_ROI_COLUMN_NAME = 'jpg_ROI_img_path'
PATHOLOGY_COLUMN_NAME = 'pathology'
CASE_TYPE_COLUMN_NAME = 'case_type'

# Model & Training Parameters
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 500
FINE_TUNE_EPOCHS = 100
LEARNING_RATE = 1e-4
RANDOM_STATE = 42

PATIENCE_EARLY_STOPPING = 25
PATIENCE_REDUCE_LR = 10

PATIENCE_EARLY_STOPPING_FT = 20
PATIENCE_REDUCE_LR_FT = 10

OUTPUT_DIR = os.path.join('./', f"run_{MODEL_NAME}_{IMG_WIDTH}_{BATCH_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"All output will be saved to: {os.path.abspath(OUTPUT_DIR)}")

# --- End of Configuration & Constants ---

# %%
# --- [The data loading and path finding sections remain the same as your original script] ---
# ... (Assuming this part runs successfully as in your script)
print("--- Initial Path Configuration Debug ---")
print(f"Current working directory (CWD): {os.getcwd()}")
print(f"BASE_DATASET_PATH (relative from CWD as defined): {BASE_DATASET_PATH}")
print(f"CALC_METADATA_CSV_PATH (relative from CWD as defined): {CALC_METADATA_CSV_PATH}") 
print(f"MASS_METADATA_CSV_PATH (relative from CWD as defined): {MASS_METADATA_CSV_PATH}")   # ADDED
print(f"IMAGE_ROOT_DIR (relative from CWD as defined): {IMAGE_ROOT_DIR}")
print(f"ACTUAL_IMAGE_FILES_BASE_DIR (relative from CWD as defined): {ACTUAL_IMAGE_FILES_BASE_DIR}")

# Resolve to absolute paths for clarity and checking
abs_base_dataset_path = os.path.abspath(BASE_DATASET_PATH)
abs_calc_metadata_csv_path = os.path.abspath(CALC_METADATA_CSV_PATH) 
abs_mass_metadata_csv_path = os.path.abspath(MASS_METADATA_CSV_PATH)   # ADDED
abs_image_root_dir = os.path.abspath(IMAGE_ROOT_DIR)
abs_actual_image_files_base_dir = os.path.abspath(ACTUAL_IMAGE_FILES_BASE_DIR)

print(f"\nResolved BASE_DATASET_PATH to absolute: {abs_base_dataset_path}")
print(f"  -> Exists? {os.path.exists(abs_base_dataset_path)} | Is Dir? {os.path.isdir(abs_base_dataset_path)}")

print(f"Resolved CALC_METADATA_CSV_PATH to absolute: {abs_calc_metadata_csv_path}") 
print(f"  -> Exists? {os.path.exists(abs_calc_metadata_csv_path)} | Is File? {os.path.isfile(abs_calc_metadata_csv_path)}")

print(f"Resolved MASS_METADATA_CSV_PATH to absolute: {abs_mass_metadata_csv_path}")   # ADDED
print(f"  -> Exists? {os.path.exists(abs_mass_metadata_csv_path)} | Is File? {os.path.isfile(abs_mass_metadata_csv_path)}")

print(f"Resolved IMAGE_ROOT_DIR to absolute: {abs_image_root_dir}")
print(f"  -> Exists? {os.path.exists(abs_image_root_dir)} | Is Dir? {os.path.isdir(abs_image_root_dir)}")

print(f"Resolved ACTUAL_IMAGE_FILES_BASE_DIR (where series folders should be): {abs_actual_image_files_base_dir}")
print(f"  -> Exists? {os.path.exists(abs_actual_image_files_base_dir)} | Is Dir? {os.path.isdir(abs_actual_image_files_base_dir)}")

if os.path.exists(abs_actual_image_files_base_dir) and os.path.isdir(abs_actual_image_files_base_dir):
    print(f"\nSample contents of ACTUAL_IMAGE_FILES_BASE_DIR ('{abs_actual_image_files_base_dir}') (first 10 items):")
    try:
        sample_contents = os.listdir(abs_actual_image_files_base_dir)[:10]
        if not sample_contents:
            print("    -> Directory is empty or unreadable.")
        for item_idx, item in enumerate(sample_contents):
            item_abs_path = os.path.join(abs_actual_image_files_base_dir, item)
            item_type = "Dir" if os.path.isdir(item_abs_path) else "File" if os.path.isfile(item_abs_path) else "Other"
            print(f"    -> [{item_type}] {item}")
    except Exception as e:
        print(f"    -> Could not list directory contents: {e}")
else:
    print("\nCRITICAL WARNING: ACTUAL_IMAGE_FILES_BASE_DIR does not exist or is not a directory. Path searches will fail.")
print("--- End of Initial Path Configuration Debug ---\n")



print("Proceeding with CSV loading...")
loaded_dfs = []

# Load Calc cases
if os.path.exists(abs_calc_metadata_csv_path):
    try:
        calc_df = pd.read_csv(abs_calc_metadata_csv_path)
        calc_df[CASE_TYPE_COLUMN_NAME] = 'calc' # Add case type identifier
        loaded_dfs.append(calc_df)
        print(f"Successfully loaded and tagged {len(calc_df)} rows from {CALC_METADATA_CSV_PATH}")
    except Exception as e:
        print(f"An error occurred while loading the CALC CSV ({CALC_METADATA_CSV_PATH}): {e}")
else:
    print(f"WARNING: CALC CSV file not found at {abs_calc_metadata_csv_path}. Skipping.")

# Load Mass cases
if os.path.exists(abs_mass_metadata_csv_path):
    try:
        mass_df = pd.read_csv(abs_mass_metadata_csv_path)
        mass_df[CASE_TYPE_COLUMN_NAME] = 'mass' # Add case type identifier
        loaded_dfs.append(mass_df)
        print(f"Successfully loaded and tagged {len(mass_df)} rows from {MASS_METADATA_CSV_PATH}")
    except Exception as e:
        print(f"An error occurred while loading the MASS CSV ({MASS_METADATA_CSV_PATH}): {e}")
else:
    print(f"WARNING: MASS CSV file not found at {abs_mass_metadata_csv_path}. Skipping.")

if not loaded_dfs:
    print("ERROR: No CSV files were loaded. Cannot proceed.")
    raise FileNotFoundError("Neither Calc nor Mass CSV files could be loaded. Check paths and file existence.")

source_df = pd.concat(loaded_dfs, ignore_index=True)
print(f"Combined DataFrame created with {len(source_df)} total rows from {len(loaded_dfs)} CSV file(s).")
print(f"Columns available in combined DataFrame: {source_df.columns.tolist()}")


# Clean and filter initial dataframe
if CONCEPTUAL_ROI_COLUMN_NAME not in source_df.columns or PATHOLOGY_COLUMN_NAME not in source_df.columns:
    print(f"ERROR: Required columns for metadata ('{CONCEPTUAL_ROI_COLUMN_NAME}' or '{PATHOLOGY_COLUMN_NAME}') not in combined CSV.")
    print(f"Available columns are: {source_df.columns.tolist()}")
    raise KeyError("Missing essential columns in combined CSV.")

source_df.dropna(subset=[CONCEPTUAL_ROI_COLUMN_NAME, PATHOLOGY_COLUMN_NAME], inplace=True)
source_df = source_df[source_df[PATHOLOGY_COLUMN_NAME].isin(['MALIGNANT', 'BENIGN'])]
print(f"Rows after initial cleaning (dropna on conceptual ROI/pathology, pathology filter): {len(source_df)}")

if source_df.empty:
    raise ValueError("Combined DataFrame is empty after initial cleaning. Cannot proceed.")

def heuristic_find_image_path(row, actual_images_root_dir_abs):
    try:
        patient_id = row['patient_id']
        breast_side = row['left or right breast']
        image_view = row['image view']
        abnormality_id = str(row['abnormality id']) # Ensure it's a string for concatenation

        csv_conceptual_roi_path = str(row.get(CONCEPTUAL_ROI_COLUMN_NAME, "")).strip()

        case_type_folder_prefix = ""
        if csv_conceptual_roi_path.startswith("jpg_img/"):
            path_part = csv_conceptual_roi_path.split('/')[1] # e.g., "Calc_Training_P_00005_..." or "Mass_Test_P_00001_..."
            # Extract the part before patient_id
            # The heuristic already includes Mass_Training and Mass_Test
            if path_part.startswith("Calc_Training_"): case_type_folder_prefix = "Calc_Training"
            elif path_part.startswith("Calc_Test_"): case_type_folder_prefix = "Calc_Test"
            elif path_part.startswith("Mass_Training_"): case_type_folder_prefix = "Mass_Training"
            elif path_part.startswith("Mass_Test_"): case_type_folder_prefix = "Mass_Test"

        if not case_type_folder_prefix:
            # print(f"DEBUG (heuristic): Could not determine case_type_folder_prefix for {patient_id} from '{csv_conceptual_roi_path}'")
            return None

        # Form search pattern for directories: e.g., /path/to/jpg_img/Calc_Training_P_00005_RIGHT_CC_1-*
        dir_search_prefix = f"{case_type_folder_prefix}_{patient_id}_{breast_side}_{image_view}_{abnormality_id}"
        full_dir_search_pattern = os.path.join(actual_images_root_dir_abs, f"{dir_search_prefix}-*")

        potential_series_dirs = glob.glob(full_dir_search_pattern)

        if not potential_series_dirs:
            # print(f"DEBUG (heuristic): No series directory found for {patient_id} with pattern '{full_dir_search_pattern}'")
            return None

        roi_filename_patterns = [
            "ROI-mask-images-img_0-*.jpg", "ROI-mask-images-img_1-*.jpg", "ROI-mask-images-img_*-*.jpg"
        ]

        for series_dir_on_disk in sorted(potential_series_dirs): # Sort to get a consistent choice if multiple match
            if os.path.isdir(series_dir_on_disk):
                for pattern in roi_filename_patterns:
                    image_search_glob = os.path.join(series_dir_on_disk, pattern)
                    found_roi_files = glob.glob(image_search_glob)
                    if found_roi_files:
                        found_roi_files.sort() # Sort to get a consistent choice
                        return found_roi_files[0] # Return the first valid ROI found
        return None
    except Exception as e:
        # print(f"DEBUG (heuristic): Error for row {row.get('patient_id', 'Unknown')} ({row.get(CASE_TYPE_COLUMN_NAME, 'N/A')} case): {e}")
        return None

print("Attempting HEURISTIC search for valid ROI paths for each CSV entry...")
source_df['full_image_path'] = source_df.apply(
    lambda r: heuristic_find_image_path(r, abs_actual_image_files_base_dir), axis=1
)

# All columns from source_df (including 'case_type' and any other original metadata)
# will be carried into metadata_df for rows where an image path was found.
metadata_df = source_df.dropna(subset=['full_image_path']).copy()
found_image_count = len(metadata_df)
print(f"Found {found_image_count} actual image files (ROIs if available) after HEURISTIC search from combined data.")
print(f"Breakdown by case type (if available in metadata_df): \n{metadata_df[CASE_TYPE_COLUMN_NAME].value_counts()}")


if found_image_count == 0:
    print("CRITICAL ERROR: Still no valid image files found even after heuristic search from combined data.")
    raise FileNotFoundError("No usable image files found even with heuristic search from combined data.")

metadata_df.rename(columns={'full_image_path': 'full_roi_path'}, inplace=True)

label_encoder = LabelEncoder()
# Ensure 'pathology_encoded' is created correctly on the copied DataFrame slice
metadata_df.loc[:, 'pathology_encoded'] = label_encoder.fit_transform(metadata_df[PATHOLOGY_COLUMN_NAME])
target_names = list(label_encoder.classes_)

# X will contain the image paths, y will contain the encoded labels.
# All other metadata columns (like 'patient_id', 'case_type', etc.) remain in metadata_df
# and can be used for deeper analysis or if a multi-input model is developed later.
X = metadata_df['full_roi_path']
y = metadata_df['pathology_encoded']
print(f"Number of samples going into train_test_split: {len(X)}")

if len(X) == 0:
     raise ValueError("Dataset is empty, cannot split.")

# Stratify by y to ensure balanced splits, especially important if classes are imbalanced
# or if combining datasets leads to different proportions.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train_val # Stratify this split too
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Print class distribution in each set to verify stratification
print(f"Train labels distribution: {np.bincount(y_train)}")
print(f"Validation labels distribution: {np.bincount(y_val)}")
print(f"Test labels distribution: {np.bincount(y_test)}")


def load_image(image_path_tensor, label_tensor):
    image_path_str = image_path_tensor.numpy().decode('utf-8')
    try:
        img = cv2.imread(image_path_str)
        if img is None: # Check if image loading failed
            dummy_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            error_label = np.int32(-1) # Special label to indicate a problem
            return dummy_img, error_label

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img, label_tensor.numpy().astype(np.int32)
    except Exception as e:
        dummy_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        error_label = np.int32(-1)
        return dummy_img, error_label

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(factor=0.1),
    tf.keras.layers.RandomContrast(factor=0.1)
])

def create_tf_dataset(image_paths, labels, batch_size, augment=False):
    image_paths_list = list(image_paths)
    labels_list = list(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths_list, labels_list))

    # Step 1: Load images
    dataset = dataset.map(lambda x, y: tf.py_function(
        load_image,
        [x, y],
        [tf.uint8, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE)

    # Step 2: Filter failed loads
    dataset = dataset.filter(lambda img, label: label != -1)

    # Step 3: Set tensor shapes
    def set_shape(img, label):
        img.set_shape((IMG_WIDTH, IMG_HEIGHT, 3))
        label.set_shape(())
        return img, label
    dataset = dataset.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)

    # Cast to float32
    dataset = dataset.map(lambda img, label: (tf.cast(img, tf.float32), label),
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.map(lambda img, label: (img, tf.expand_dims(label, axis=-1)),
                          num_parallel_calls=tf.data.AUTOTUNE)
    # Batch before augmentation
    dataset = dataset.batch(batch_size)

    # Step 4: Apply augmentation with keyword argument
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),  # Use keyword
                              num_parallel_calls=tf.data.AUTOTUNE)

    # Step 5: Apply preprocessing
    dataset = dataset.map(lambda x, y: (tf.keras.applications.efficientnet.preprocess_input(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Step 6: Prefetch
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def visualize_augmentations(sample_image_paths, augmentation_layer, output_dir):
    print("Generating and saving data augmentation visualization...")
    num_examples = len(sample_image_paths)
    num_augmentations = 4 # How many augmented versions to show for each original
    
    plt.figure(figsize=(5 * num_augmentations, 5 * num_examples))
    
    for i, image_path in enumerate(sample_image_paths):
        # Load the original image using a simplified version of the preprocessing
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Display the original image
        ax = plt.subplot(num_examples, num_augmentations + 1, i * (num_augmentations + 1) + 1)
        plt.imshow(img)
        plt.title(f"Original {i+1}")
        plt.axis("off")
        
        # Add batch dimension and apply augmentations
        img_tensor = tf.expand_dims(tf.convert_to_tensor(img), 0)
        for j in range(num_augmentations):
            augmented_image = augmentation_layer(img_tensor, training=True)
            ax = plt.subplot(num_examples, num_augmentations + 1, i * (num_augmentations + 1) + j + 2)
            plt.imshow(tf.squeeze(augmented_image, axis=0).numpy().astype("uint8"))
            plt.title(f"Augmented {j+1}")
            plt.axis("off")
            
    plt.tight_layout()
    # Save the figure to the specified output directory
    save_path = os.path.join(output_dir, "data_augmentation_examples.png")
    plt.savefig(save_path)
    print(f"Augmentation visualization saved to {save_path}")
    plt.show()

# --- Call the new visualization function ---
# Take a few sample images from the training set to show
num_viz_samples = 3
if len(X_train) >= num_viz_samples:
    visualize_augmentations(
        sample_image_paths=X_train.iloc[:num_viz_samples],
        augmentation_layer=data_augmentation,
        output_dir=OUTPUT_DIR
    )

print("Recreating TensorFlow datasets with updated image loading logic...")
train_dataset = create_tf_dataset(X_train, y_train, BATCH_SIZE, augment=True)
val_dataset = create_tf_dataset(X_val, y_val, BATCH_SIZE, augment=False)
test_dataset = create_tf_dataset(X_test, y_test, BATCH_SIZE, augment=False)

print("Verifying dataset integrity (this might take a moment)...")
train_batches = 0
train_samples_effective = 0
for images, labels in train_dataset:
    train_batches += 1
    train_samples_effective += labels.shape[0]
print(f"Number of batches in train_dataset: {train_batches}")
print(f"Effective number of samples in train_dataset after filtering: {train_samples_effective}")

if train_batches > 0:
    for images, labels in train_dataset.take(1):
        print("Sample batch shape from train_dataset:", images.shape, labels.shape)
else:
    print("Warning: train_dataset is empty after filtering. Check for widespread image loading issues.")

val_batches = 0
val_samples_effective = 0
for images, labels in val_dataset:
    val_batches +=1
    val_samples_effective += labels.shape[0]
print(f"Number of batches in val_dataset: {val_batches}")
print(f"Effective number of samples in val_dataset after filtering: {val_samples_effective}")


test_batches = 0
test_samples_effective = 0
for images, labels in test_dataset:
    test_batches += 1
    test_samples_effective += labels.shape[0]
print(f"Number of batches in test_dataset: {test_batches}")
print(f"Effective number of samples in test_dataset after filtering: {test_samples_effective}")


# Check if any dataset is empty, which could cause issues during training/evaluation
if train_samples_effective == 0 or val_samples_effective == 0:
    print("CRITICAL WARNING: Training or Validation dataset is empty after processing. Model training cannot proceed effectively.")
    # Depending on the severity, you might want to raise an error here
    # raise ValueError("Training or Validation dataset is empty.")


# %%
# --- 2. Model Architecture ---
print("\nPhase 2: Building the Model")
base_model = EfficientNetB4(include_top=False, weights='imagenet',
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
base_model.trainable = False # Start with base model frozen

inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = Dropout(0.3, name="top_dropout_1")(x)
x = Dense(128, activation='relu', name="dense_128")(x)
x = Dropout(0.3, name="top_dropout_2")(x)
outputs = Dense(1, activation='sigmoid', name="predictions")(x)
model = Model(inputs, outputs)

# %%
# --- 3. Model Compilation ---
print("\nPhase 3: Compiling the Model")
optimizer = Adam(learning_rate=LEARNING_RATE)

# MODIFIED: Adjust loss based on number of classes
if len(target_names) <= 2: # Binary classification (or single class if an error, but usually benign/malignant)
    print("Binary classification")
    loss = tf.keras.losses.BinaryCrossentropy()
    # For binary, ensure 'accuracy' is suitable. AUC, Precision, Recall are fine.
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc'),
               tf.keras.metrics.Precision(name='precision'), 
               tf.keras.metrics.Recall(name='recall'), 
               tf.keras.metrics.F1Score(name='f1_score'),
               tf.keras.metrics.FalseNegatives(name='false_negatives'),
               tf.keras.metrics.FalsePositives(name='false_positives')]
else: # Multiclass classification
    loss = tf.keras.losses.SparseCategoricalCrossentropy() # Assuming y_train, etc., are integer labels
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC might need multi_label=True or specific setup for multiclass
    # For multiclass, typical metrics are accuracy, sparse_categorical_accuracy.
    # Precision and Recall can be more complex (e.g., weighted, macro).
    # For simplicity, starting with accuracy and AUC.

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

print("Available metrics names:", model.metrics_names)

# %%
print("\nPhase 4: Training the Model (Head Only)")
# Ensure datasets are not empty before starting training
if train_samples_effective == 0 or val_samples_effective == 0:
    print("ERROR: Cannot start head training because train or validation dataset is empty.")
else:

    checkpoint_filepath_head = os.path.join(OUTPUT_DIR, 'best_model_head_only.keras')
    callbacks_head = [
        ModelCheckpoint(filepath=checkpoint_filepath_head, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=PATIENCE_EARLY_STOPPING, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_REDUCE_LR, min_lr=1e-7, mode='min')
    ]
    history_head = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_head
    )
    print("Loading best weights from head training...")
    if os.path.exists(checkpoint_filepath_head):
        model.load_weights(checkpoint_filepath_head)
    else:
        print(f"Warning: Checkpoint file {checkpoint_filepath_head} not found. Using last model weights.")


# --- 4b. Fine-tuning Phase ---
print("\nPhase 4b: Fine-tuning (Unfreezing some base model layers)")
base_model.trainable = True

# Unfreeze layers from a certain block onwards in EfficientNetB4
# EfficientNetB4 has blocks named 'block2a_expand_conv', 'block3a_expand_conv', ..., 'block7a_expand_conv'
# We can choose to unfreeze from 'block6a' or 'block5a' onwards
# For this example, let's unfreeze from 'block5a' onwards.
# You might need to inspect base_model.summary() to choose the right layers.

# Fine-tuning strategy: Unfreeze more layers
# Set base_model.trainable = True first
# Then, selectively re-freeze earlier layers if desired
# For EfficientNet, it's common to unfreeze the top blocks.

fine_tune_at_layer_name = 'block6a_expand_conv' # Here we are just unfrezzing one
set_trainable = False
for layer in base_model.layers:
    if layer.name == fine_tune_at_layer_name:
        set_trainable = True
    if set_trainable:
        if not isinstance(layer, tf.keras.layers.BatchNormalization): # Keep BN frozen
            layer.trainable = True
        else:
            layer.trainable = False # Explicitly keep BN frozen
    else:
        layer.trainable = False


optimizer_fine_tune = Adam(learning_rate=LEARNING_RATE / 10) # Use a smaller LR
model.compile(optimizer=optimizer_fine_tune, loss=loss, metrics=metrics) # Re-compile
model.summary() # Show summary with new trainable params
if train_samples_effective == 0 or val_samples_effective == 0:
     print("ERROR: Cannot start fine-tuning because train or validation dataset is empty.")
else:

    checkpoint_filepath_finetune = os.path.join(OUTPUT_DIR, 'best_model_finetuned.keras')
    callbacks_finetune = [
        ModelCheckpoint(filepath=checkpoint_filepath_finetune, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=PATIENCE_EARLY_STOPPING_FT, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE_REDUCE_LR_FT, min_lr=1e-8, mode='min')
    ]

    # Determine initial epoch for fine-tuning
    initial_fine_tune_epoch = 0
    if 'history_head' in locals() and hasattr(history_head, 'epoch') and history_head.epoch:
        initial_fine_tune_epoch = history_head.epoch[-1] + 1
    else: # If head training was skipped or history is unavailable
        initial_fine_tune_epoch = 0 # Or EPOCHS if you want to assume head training ran for all its epochs
        EPOCHS = 0 # Ensure we don't re-run head training if it was skipped

    history_fine_tune = model.fit(
        train_dataset,
        epochs=EPOCHS + FINE_TUNE_EPOCHS, # Total epochs
        initial_epoch=initial_fine_tune_epoch, # Continue from where head training left off
        validation_data=val_dataset,
        callbacks=callbacks_finetune
    )
    print("Loading best weights from fine-tuning...")
    if os.path.exists(checkpoint_filepath_finetune):
        model.load_weights(checkpoint_filepath_finetune)
    else:
        print(f"Warning: Checkpoint file {checkpoint_filepath_finetune} not found. Using last model weights from fine-tuning.")



# %%
# --- 5. Model Evaluation ---
print("\nPhase 5: Evaluating the Model on Test Set")

if test_samples_effective == 0:
    print("ERROR: Test dataset is empty. Cannot evaluate model.")
    test_accuracy = 0
    test_auc = 0
else:
    results = model.evaluate(test_dataset, verbose=1)
    
    # The F1-score is at index 5, so we need at least 6 items in results
    print(f"results: {results}")
    if len(results) >= 6:
        final_loss = results[0]
        final_acc = results[1]
        final_auc = results[2]
        final_precision = results[3]
        final_recall = results[4]
        final_f1_score = results[5] # <-- Extract the F1-score here

        print(f"Final Loss: {final_loss}")
        print(f"Final Accuracy: {final_acc}")
        print(f"Final AUC: {final_auc}")
        print(f"Final Precision: {final_precision}")
        print(f"Final Recall: {final_recall}")
        print(f"Final F1-Score: {final_f1_score}") # <-- Print it for confirmation

        # MODIFIED: Update the filename to include loss and F1-score
        history_plot_filename = f"training_history_Loss{final_loss:.3f}_Acc{final_acc:.3f}_AUC{final_auc:.3f}_F1{final_f1_score:.3f}_Loss{final_loss}.png"
    else:
        print("Error: Not enough metrics returned from model.evaluate to extract F1-score.")
        # Fallback filename if F1-score isn't available
        final_acc = results[1]
        final_auc = results[2]
        history_plot_filename = f"training_history_Acc{final_acc:.3f}_AUC{final_auc:.3f}.png"


    print("\nFull evaluation results:", results)
    print("Model metrics names:", model.metrics_names)
    y_pred_proba = model.predict(test_dataset)

    # Extract true labels correctly, regardless of whether test_dataset was batched
    y_true_test = []
    for _, labels_batch in test_dataset.unbatch().batch(BATCH_SIZE): # Re-batch after unbatching to iterate easily
        y_true_test.extend(labels_batch.numpy())
    y_true_test = np.array(y_true_test)

    if len(target_names) <= 2: # Binary classification
        y_pred_classes = (y_pred_proba > 0.5).astype(int).flatten()
    else: # Multiclass classification
        y_pred_classes = np.argmax(y_pred_proba, axis=1)


    if len(y_true_test) == 0:
        print("Warning: No true labels extracted from the test set. Cannot generate classification report or confusion matrix.")
    elif len(y_true_test) != len(y_pred_classes):
         print(f"Warning: Mismatch in number of true labels ({len(y_true_test)}) and predicted classes ({len(y_pred_classes)}). Skipping report/matrix.")
    else:
        print("\nClassification Report (Test Set):")
        print(classification_report(y_true_test, y_pred_classes, target_names=target_names, labels=range(len(target_names))))


        cm = confusion_matrix(y_true_test, y_pred_classes, labels=range(len(target_names)))
        print("\nConfusion Matrix (Test Set):")
        print(cm)

        plt.figure(figsize=(8,8)) # Made figure a bit bigger
        ax = plt.gca()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title('Confusion Matrix (Test Set)', fontsize=18)
        
    
        cm_save_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
        plt.savefig(cm_save_path)
        print(f"Confusion matrix saved to {cm_save_path}")
        plt.show()


acc, val_acc, loss_hist, val_loss_hist, auc, val_auc = [], [], [], [], [], []
f1, val_f1 = [], []
false_positives, val_false_positives = [], []
false_negatives, val_false_negatives = [], []

epochs_range_head_len = 0
print(f"history_head.history: {history_head.history}")
if 'history_head' in locals() and hasattr(history_head, 'history'):
    acc.extend(history_head.history.get('accuracy', []))
    val_acc.extend(history_head.history.get('val_accuracy', []))
    loss_hist.extend(history_head.history.get('loss', []))
    val_loss_hist.extend(history_head.history.get('val_loss', []))
    auc.extend(history_head.history.get('auc', []))
    val_auc.extend(history_head.history.get('val_auc', []))
    epochs_range_head_len = len(history_head.history.get('accuracy', []))
    f1.extend(history_head.history.get('f1_score', []))
    val_f1.extend(history_head.history.get('val_f1_score', []))
    false_positives.extend(history_head.history.get('false_positives', []))
    val_false_positives.extend(history_head.history.get('val_false_positives', []))
    false_negatives.extend(history_head.history.get('false_negatives', []))
    val_false_negatives.extend(history_head.history.get('val_false_negatives', []))

if 'history_fine_tune' in locals() and hasattr(history_fine_tune, 'history'):
    acc.extend(history_fine_tune.history.get('accuracy', []))
    val_acc.extend(history_fine_tune.history.get('val_accuracy', []))
    loss_hist.extend(history_fine_tune.history.get('loss', []))
    val_loss_hist.extend(history_fine_tune.history.get('val_loss', []))
    auc.extend(history_fine_tune.history.get('auc', []))
    val_auc.extend(history_fine_tune.history.get('val_auc', []))
    f1.extend(history_fine_tune.history.get('f1_score', []))
    val_f1.extend(history_fine_tune.history.get('val_f1_score', []))
    false_positives.extend(history_fine_tune.history.get('false_positives', []))
    val_false_positives.extend(history_fine_tune.history.get('val_false_positives', []))
    false_negatives.extend(history_fine_tune.history.get('false_negatives', []))
    val_false_negatives.extend(history_fine_tune.history.get('val_false_negatives', []))

epochs_range_total = range(len(acc))

if epochs_range_total: # Only plot if there's history
    plt.figure(figsize=(24, 16)) # Made figure wider
    
    # Plot 1: Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range_total, acc, label='Training Accuracy')
    plt.plot(epochs_range_total, val_acc, label='Validation Accuracy')
    if epochs_range_head_len > 0 and epochs_range_head_len < len(epochs_range_total):
        plt.axvline(x=epochs_range_head_len-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot 2: Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range_total, loss_hist, label='Training Loss')
    plt.plot(epochs_range_total, val_loss_hist, label='Validation Loss')
    if epochs_range_head_len > 0 and epochs_range_head_len < len(epochs_range_total):
        plt.axvline(x=epochs_range_head_len-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Plot 3: AUC
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range_total, auc, label='Training AUC')
    plt.plot(epochs_range_total, val_auc, label='Validation AUC')
    if epochs_range_head_len > 0 and epochs_range_head_len < len(epochs_range_total):
        plt.axvline(x=epochs_range_head_len-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')

    # Plot 4: F1-Score
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range_total, f1, label='Training F1')
    plt.plot(epochs_range_total, val_f1, label='Validation F1')
    if epochs_range_head_len > 0 and epochs_range_head_len < len(epochs_range_total):
        plt.axvline(x=epochs_range_head_len-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation F1-Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')

    # Plot 5: False Positives
    plt.subplot(2, 3, 5)
    plt.plot(epochs_range_total, false_positives, label='Training False Positives')
    plt.plot(epochs_range_total, val_false_positives, label='Validation False Positives')
    if epochs_range_head_len > 0 and epochs_range_head_len < len(epochs_range_total):
        plt.axvline(x=epochs_range_head_len-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation False Positives')
    plt.xlabel('Epochs')
    plt.ylabel('Count')
    plt.yscale('log')  # Log scale for better visualization

    # Plot 6: False Negatives
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range_total, false_negatives, label='Training False Negatives')
    plt.plot(epochs_range_total, val_false_negatives, label='Validation False Negatives')
    if epochs_range_head_len > 0 and epochs_range_head_len < len(epochs_range_total):
        plt.axvline(x=epochs_range_head_len-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation False Negatives')
    plt.xlabel('Epochs')
    plt.ylabel('Count')
    plt.yscale('log')  # Log scale for better visualization

    plt.tight_layout()
    

    #print("Available metrics names:", model.metrics_names)
    # Use the metrics already extracted in the evaluation section
    final_acc = results[1]  # Accuracy
    final_auc = results[2]  # AUC
    history_plot_filename = f"training_history{final_acc:.3f}_AUC{final_auc:.3f}_F1{final_f1_score:.3f}_Loss{final_loss}.png"
    history_save_path = os.path.join(OUTPUT_DIR, history_plot_filename)
    plt.savefig(history_save_path)
    print(f"Training history plot saved to {history_save_path}")
    
    plt.show()
else:
    print("No training history found to plot.")

print("\n--- End of Training ---")


