# SECTION 1: INSTALLATION AND IMPORTS


# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import requests
from PIL import Image
import zipfile
import shutil
from google.colab import drive, files
import random
import warnings
from glob import glob
from collections import defaultdict
import time
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Set mixed precision for faster training
try:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled for faster training")
except Exception as e:
    print(f"Mixed precision not available: {e}")

# SECTION 2: ENHANCED DATASET PREPARATION

# Mount Google Drive
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")
except Exception as e:
    print(f"Google Drive mount failed: {e}")

# Create directories for datasets
os.makedirs('/content/datasets', exist_ok=True)
os.makedirs('/content/prepared_data/organic', exist_ok=True)
os.makedirs('/content/prepared_data/inorganic', exist_ok=True)

# Upload Kaggle API key
def setup_kaggle():
    try:
        print("Please upload your kaggle.json file when prompted:")
        uploaded = files.upload()

        # Setup Kaggle API
        os.makedirs('/root/.kaggle', exist_ok=True)
        os.rename('/content/kaggle.json', '/root/.kaggle/kaggle.json')
        os.chmod('/root/.kaggle/kaggle.json', 600)
        print("Kaggle API configured successfully")
        return True
    except Exception as e:
        print(f"Kaggle setup failed: {e}")
        print("Please ensure kaggle.json is uploaded correctly")
        return False

# Download datasets with enhanced error handling and progress tracking
def download_datasets_enhanced():
    try:
        print("Downloading datasets with maximum data extraction...")

        # Expanded list of relevant datasets for maximum data utilization
        datasets_to_download = [
            ("moltean/fruits", "fruits.zip"),
            ("mostafaabla/garbage-classification", "garbage-classification.zip"),
            ("utkarshsaxenadn/fruits-classification", "fruits-classification.zip"),
            ("sriramr/fruits-fresh-and-rotten-for-classification", "fruits-fresh-rotten.zip"),
            ("chrisfilo/fruit-recognition", "fruit-recognition.zip"),
            ("kritikseth/fruit-and-vegetable-image-recognition", "fruit-vegetable-recognition.zip"),
        ]

        downloaded_count = 0
        for dataset, filename in datasets_to_download:
            try:
                print(f"Downloading {dataset}...")
                os.system(f'kaggle datasets download -d {dataset} -p /content/datasets/')
                if os.path.exists(f'/content/datasets/{filename}'):
                    print(f"✅ Downloaded {dataset}")
                    downloaded_count += 1
                else:
                    print(f"⚠️ Download may have failed for {dataset}")
            except Exception as e:
                print(f"Failed to download {dataset}: {e}")

        print(f"Successfully downloaded {downloaded_count}/{len(datasets_to_download)} datasets")
        return downloaded_count > 0
    except Exception as e:
        print(f"Dataset download failed: {e}")
        return False

# Enhanced extraction with progress tracking
def extract_datasets_enhanced():
    try:
        print("Extracting datasets with enhanced discovery...")

        # Find all zip files in datasets directory
        zip_files = glob('/content/datasets/*.zip')
        print(f"Found {len(zip_files)} zip files to extract")

        extracted_count = 0
        for zip_path in zip_files:
            filename = os.path.basename(zip_path)
            extract_dir = os.path.splitext(zip_path)[0]

            try:
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"✅ Extracted {filename} to {extract_dir}")
                extracted_count += 1
            except Exception as e:
                print(f"Failed to extract {filename}: {e}")

        print(f"Successfully extracted {extracted_count}/{len(zip_files)} datasets")
        return extracted_count > 0
    except Exception as e:
        print(f"Dataset extraction failed: {e}")
        return False

# Setup and download datasets
if setup_kaggle():
    download_datasets_enhanced()
    extract_datasets_enhanced()

# SECTION 3: INTELLIGENT DATA DISCOVERY AND ORGANIZATION

def discover_all_image_data():
    """
    Enhanced data discovery function to find ALL available images
    and categorize them intelligently for maximum data utilization
    """
    print("\n" + "="*60)
    print("INTELLIGENT DATA DISCOVERY")
    print("="*60)

    # Define keywords for organic classification
    organic_keywords = {
        'fruits': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'mango', 'pineapple', 'watermelon', 'kiwi', 'peach', 'pear', 'cherry', 'lemon', 'lime', 'avocado', 'coconut', 'papaya', 'fresh', 'fruit'],
        'vegetables': ['tomato', 'potato', 'carrot', 'onion', 'garlic', 'pepper', 'broccoli', 'cabbage', 'lettuce', 'spinach', 'cucumber', 'corn', 'bean', 'celery', 'vegetable'],
        'organic_waste': ['organic', 'compost', 'food', 'kitchen', 'peel', 'scrap', 'waste', 'leftover', 'rotten'],
        'biodegradable': ['paper', 'cardboard', 'newspaper', 'tissue']
    }

    # Define keywords for inorganic classification
    inorganic_keywords = {
        'plastic': ['plastic', 'bottle', 'bag', 'container', 'cup', 'plate', 'wrapper', 'packaging'],
        'metal': ['metal', 'aluminum', 'steel', 'can', 'tin', 'wire', 'screw'],
        'glass': ['glass', 'bottle', 'jar', 'window', 'mirror'],
        'textile': ['clothes', 'shirt', 'pants', 'dress', 'shoes', 'fabric'],
        'electronic': ['battery', 'electronic', 'circuit', 'phone', 'computer'],
        'other_inorganic': ['trash', 'garbage', 'litter', 'junk']
    }

    # Data discovery results
    discovered_data = {
        'organic': defaultdict(list),
        'inorganic': defaultdict(list)
    }

    total_images_found = 0

    # Scan all extracted datasets
    base_dirs = ['/content/datasets']

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        print(f"\n🔍 Scanning {base_dir} for image data...")

        # Walk through all directories
        for root, dirs, files in os.walk(base_dir):
            # Skip hidden directories and files
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            # Find image files
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

            if len(image_files) > 0:
                folder_name = os.path.basename(root).lower()
                parent_folder = os.path.basename(os.path.dirname(root)).lower()
                full_path_lower = root.lower()

                total_images_found += len(image_files)

                # Classify as organic or inorganic based on folder names and paths
                is_organic = False
                is_inorganic = False
                category = 'unknown'

                # Check for organic keywords
                for cat, keywords in organic_keywords.items():
                    for keyword in keywords:
                        if (keyword in folder_name or keyword in parent_folder or
                            keyword in full_path_lower):
                            is_organic = True
                            category = cat
                            break
                    if is_organic:
                        break

                # Check for inorganic keywords if not already classified as organic
                if not is_organic:
                    for cat, keywords in inorganic_keywords.items():
                        for keyword in keywords:
                            if (keyword in folder_name or keyword in parent_folder or
                                keyword in full_path_lower):
                                is_inorganic = True
                                category = cat
                                break
                        if is_inorganic:
                            break

                # Store discovered data
                if is_organic:
                    discovered_data['organic'][category].append({
                        'path': root,
                        'count': len(image_files),
                        'folder_name': folder_name,
                        'parent_folder': parent_folder
                    })
                    print(f"  ORGANIC [{category}]: {folder_name} ({len(image_files)} images)")
                elif is_inorganic:
                    discovered_data['inorganic'][category].append({
                        'path': root,
                        'count': len(image_files),
                        'folder_name': folder_name,
                        'parent_folder': parent_folder
                    })
                    print(f"  INORGANIC [{category}]: {folder_name} ({len(image_files)} images)")
                else:
                    # Log unclassified for manual review
                    if len(image_files) > 10:  # Only log folders with significant image count
                        print(f"  UNCLASSIFIED: {folder_name} ({len(image_files)} images)")

    print(f"\nDISCOVERY SUMMARY:")
    print(f"Total images found: {total_images_found:,}")

    # Print detailed breakdown
    organic_total = 0
    for category, folders in discovered_data['organic'].items():
        category_total = sum([folder['count'] for folder in folders])
        organic_total += category_total
        print(f"ORGANIC {category}: {category_total:,} images in {len(folders)} folders")

    inorganic_total = 0
    for category, folders in discovered_data['inorganic'].items():
        category_total = sum([folder['count'] for folder in folders])
        inorganic_total += category_total
        print(f"INORGANIC {category}: {category_total:,} images in {len(folders)} folders")

    print(f"Classified: {organic_total + inorganic_total:,} images")
    print(f"Unclassified: {total_images_found - organic_total - inorganic_total:,} images")
    print(f"Classification rate: {((organic_total + inorganic_total) / total_images_found * 100):.1f}%")

    return discovered_data, total_images_found

def smart_data_sampling(discovered_data, target_samples_per_class=50000):
    """
    Intelligent sampling to balance classes and maximize data utilization
    """
    print(f"\nSMART DATA SAMPLING - Target: {target_samples_per_class:,} per class")
    print("="*60)

    sampling_strategy = {
        'organic': {},
        'inorganic': {}
    }

    for class_type in ['organic', 'inorganic']:
        total_available = sum([
            sum([folder['count'] for folder in folders])
            for folders in discovered_data[class_type].values()
        ])

        print(f"\n{class_type.upper()} CLASS SAMPLING:")
        print(f"Total available: {total_available:,} images")

        if total_available <= target_samples_per_class:
            # Use all available data
            print(f"Strategy: Use ALL available data ({total_available:,} images)")
            for category, folders in discovered_data[class_type].items():
                for folder in folders:
                    sampling_strategy[class_type][folder['path']] = folder['count']
        else:
            # Proportional sampling with minimum guarantees
            print(f"Strategy: Proportional sampling to {target_samples_per_class:,} images")

            # Calculate proportions
            category_totals = {}
            for category, folders in discovered_data[class_type].items():
                category_totals[category] = sum([folder['count'] for folder in folders])

            # Ensure each category gets at least some representation
            min_per_category = max(1000, target_samples_per_class // len(category_totals))
            remaining_samples = target_samples_per_class

            for category, total_count in category_totals.items():
                if total_count <= min_per_category:
                    # Use all data from small categories
                    category_allocation = total_count
                else:
                    # Proportional allocation for large categories
                    proportion = total_count / total_available
                    category_allocation = max(min_per_category, int(target_samples_per_class * proportion))

                remaining_samples -= category_allocation

                print(f"  {category}: {category_allocation:,} images")

                # Distribute within category
                folders = discovered_data[class_type][category]
                category_total = sum([folder['count'] for folder in folders])

                for folder in folders:
                    if category_total <= category_allocation:
                        # Use all images from this folder
                        folder_allocation = folder['count']
                    else:
                        # Proportional sampling within category
                        folder_proportion = folder['count'] / category_total
                        folder_allocation = max(1, int(category_allocation * folder_proportion))

                    sampling_strategy[class_type][folder['path']] = folder_allocation

    return sampling_strategy

def organize_datasets_maximized():
    """
    MAXIMIZED dataset organization using intelligent discovery and sampling
    """
    print("\n" + "="*60)
    print("DATASET ORGANIZATION")
    print("="*60)

    # Discover all available data
    discovered_data, total_found = discover_all_image_data()

    # Create sampling strategy
    sampling_strategy = smart_data_sampling(discovered_data, target_samples_per_class=100000)

    organic_count = 0
    inorganic_count = 0

    # Process organic data
    print("\nPROCESSING ORGANIC DATA...")
    for source_path, target_count in sampling_strategy['organic'].items():
        if not os.path.exists(source_path):
            continue

        # Get all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            image_files.extend(glob(os.path.join(source_path, f'*{ext}')))
            image_files.extend(glob(os.path.join(source_path, f'*{ext.upper()}')))

        # Sample images
        if len(image_files) <= target_count:
            selected_files = image_files
        else:
            selected_files = random.sample(image_files, target_count)

        if len(selected_files) > 0:
            # Create destination directory
            folder_name = os.path.basename(source_path)
            parent_name = os.path.basename(os.path.dirname(source_path))
            dest_name = f"{parent_name}_{folder_name}".replace(' ', '_').replace('-', '_')
            dest_path = f'/content/prepared_data/organic/{dest_name}'

            try:
                os.makedirs(dest_path, exist_ok=True)

                # Copy selected images
                copied_count = 0
                for i, img_file in enumerate(selected_files):
                    try:
                        # Verify image can be opened
                        with Image.open(img_file) as img:
                            if img.mode not in ['RGB', 'RGBA', 'L']:
                                continue

                        # Copy image with new name to avoid conflicts
                        ext = os.path.splitext(img_file)[1]
                        new_name = f"{dest_name}_{i:06d}{ext}"
                        shutil.copy2(img_file, os.path.join(dest_path, new_name))
                        copied_count += 1

                    except Exception as e:
                        continue  # Skip corrupted images

                if copied_count > 0:
                    print(f"  {dest_name}: {copied_count:,} images")
                    organic_count += copied_count
                else:
                    shutil.rmtree(dest_path, ignore_errors=True)

            except Exception as e:
                print(f"  Error processing {folder_name}: {e}")

    # Process inorganic data
    print("\nPROCESSING INORGANIC DATA...")
    for source_path, target_count in sampling_strategy['inorganic'].items():
        if not os.path.exists(source_path):
            continue

        # Get all image files
        image_files = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            image_files.extend(glob(os.path.join(source_path, f'*{ext}')))
            image_files.extend(glob(os.path.join(source_path, f'*{ext.upper()}')))

        # Sample images
        if len(image_files) <= target_count:
            selected_files = image_files
        else:
            selected_files = random.sample(image_files, target_count)

        if len(selected_files) > 0:
            # Create destination directory
            folder_name = os.path.basename(source_path)
            parent_name = os.path.basename(os.path.dirname(source_path))
            dest_name = f"{parent_name}_{folder_name}".replace(' ', '_').replace('-', '_')
            dest_path = f'/content/prepared_data/inorganic/{dest_name}'

            try:
                os.makedirs(dest_path, exist_ok=True)

                # Copy selected images
                copied_count = 0
                for i, img_file in enumerate(selected_files):
                    try:
                        # Verify image can be opened
                        with Image.open(img_file) as img:
                            if img.mode not in ['RGB', 'RGBA', 'L']:
                                continue

                        # Copy image with new name to avoid conflicts
                        ext = os.path.splitext(img_file)[1]
                        new_name = f"{dest_name}_{i:06d}{ext}"
                        shutil.copy2(img_file, os.path.join(dest_path, new_name))
                        copied_count += 1

                    except Exception as e:
                        continue  # Skip corrupted images

                if copied_count > 0:
                    print(f"  {dest_name}: {copied_count:,} images")
                    inorganic_count += copied_count
                else:
                    shutil.rmtree(dest_path, ignore_errors=True)

            except Exception as e:
                print(f"  Error processing {folder_name}: {e}")

    print(f"\n" + "="*60)
    print("DATASET ORGANIZATION COMPLETED!")
    print("="*60)
    print(f"ORGANIC samples: {organic_count:,}")
    print(f"INORGANIC samples: {inorganic_count:,}")
    print(f"Total samples: {organic_count + inorganic_count:,}")
    print(f"Data utilization: {((organic_count + inorganic_count) / total_found * 100):.1f}% of discovered images")
    print(f"Class balance ratio: {min(organic_count, inorganic_count) / max(organic_count, inorganic_count):.2f}")

    return organic_count, inorganic_count

# Set random seed for reproducible sampling
random.seed(42)
np.random.seed(42)

# Organize datasets with maximum utilization
organic_count, inorganic_count = organize_datasets_maximized()

# SECTION 4: ENHANCED DATA PREPROCESSING WITH LARGE DATASET SUPPORT

# Optimized parameters for large datasets
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  # Increased for better GPU utilization
VALIDATION_SPLIT = 0.15  # Reduced to keep more data for training

def create_optimized_data_generators():
    """Create optimized data generators for large datasets"""

    try:
        # Enhanced training augmentation for large datasets
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.85, 1.15],
            channel_shift_range=0.1,
            fill_mode='nearest',
            validation_split=VALIDATION_SPLIT
        )

        # Validation data (only rescaling)
        validation_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=VALIDATION_SPLIT
        )

        # Create generators with optimized settings
        train_generator = train_datagen.flow_from_directory(
            '/content/prepared_data',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42,
            interpolation='bilinear'  # Faster than default
        )

        validation_generator = validation_datagen.flow_from_directory(
            '/content/prepared_data',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            shuffle=False,
            seed=42,
            interpolation='bilinear'
        )

        print("✅ Optimized data generators created successfully")
        print("📊 Class indices:", train_generator.class_indices)
        print(f"🏋️ Training samples: {train_generator.samples:,}")
        print(f"🔍 Validation samples: {validation_generator.samples:,}")
        print(f"📦 Batch size: {BATCH_SIZE}")
        print(f"🔄 Steps per epoch: {train_generator.samples // BATCH_SIZE}")

        return train_generator, validation_generator

    except Exception as e:
        print(f"❌ Error creating data generators: {e}")
        return None, None

# Create optimized data generators
train_generator, validation_generator = create_optimized_data_generators()

if train_generator is None or validation_generator is None:
    print("Failed to create data generators. Stopping execution.")
    exit()

# Calculate class weights for balanced training
try:
    y_train = train_generator.classes
    if len(np.unique(y_train)) > 1:
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        print(f"⚖️ Class weights computed: {class_weight_dict}")
    else:
        print("Only one class in training data. Using equal weights.")
        class_weight_dict = {0: 1.0, 1: 1.0}
except Exception as e:
    print(f"Error computing class weights: {e}")
    class_weight_dict = {0: 1.0, 1: 1.0}

# ============================================================================
# SECTION 5: OPTIMIZED MODEL ARCHITECTURE FOR LARGE DATASETS
# ============================================================================

def create_optimized_model():
    """Create optimized EfficientNetB0 model for large dataset training"""

    try:
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )

        # Freeze base model initially for transfer learning
        base_model.trainable = False

        # Optimized classification head for large datasets
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.4),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid', dtype='float32')
        ])

        # Compile with optimized settings
        initial_learning_rate = 0.001
        model.compile(
            optimizer=Adam(learning_rate=initial_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        print("✅ Optimized model created successfully")
        print("🧠 Architecture: EfficientNetB0 + Enhanced Classification Head")
        print(f"🔢 Total parameters: {model.count_params():,}")
        print(f"🎯 Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        return model

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return None

# Create optimized model
model = create_optimized_model()

if model is None:
    print("Failed to create model. Stopping execution.")
    exit()

model.summary()

# SECTION 6: ADVANCED TRAINING CONFIGURATION FOR LARGE DATASETS

# Create model checkpoint directory
checkpoint_dir = '/content/drive/MyDrive/optimized_waste_classifier_checkpoints'
try:
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_optimized_model.keras')
except:
    checkpoint_dir = '/content/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_optimized_model.keras')

def create_advanced_callbacks():
    """Create advanced training callbacks optimized for large datasets"""

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=5,
            min_lr=0.000001,
            verbose=1,
            min_delta=0.001
        )
    ]

    # Add model checkpoint
    try:
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )
        print(f"💾 Model checkpoint will be saved to: {checkpoint_path}")
    except Exception as e:
        print(f"Cannot create model checkpoint: {e}")

    return callbacks

def fine_tune_model(model, train_gen, val_gen, initial_epochs=30):
    """
    Two-stage training: Initial training + Fine-tuning with unfrozen layers
    """
    print("\n" + "="*80)
    print("🚀 STAGE 1: INITIAL TRAINING WITH FROZEN BASE")
    print("="*80)

    # Stage 1: Train with frozen base
    callbacks = create_advanced_callbacks()

    history_1 = model.fit(
        train_gen,
        steps_per_epoch=min(1000, train_gen.samples // BATCH_SIZE),  # Limit steps for faster initial training
        epochs=initial_epochs,
        validation_data=val_gen,
        validation_steps=min(200, val_gen.samples // BATCH_SIZE),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\n" + "="*80)
    print("🔥 STAGE 2: FINE-TUNING WITH UNFROZEN LAYERS")
    print("="*80)

    # Stage 2: Unfreeze and fine-tune
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:-50]:  # Freeze first layers
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    print(f"🎯 Trainable parameters after unfreezing: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

    # Fine-tuning training
    history_2 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        epochs=20,  # Fewer epochs for fine-tuning
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
        initial_epoch=initial_epochs
    )

    # Combine histories
    combined_history = {}
    for key in history_1.history.keys():
        combined_history[key] = history_1.history[key] + history_2.history[key]

    # Create a mock history object
    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict

    return CombinedHistory(combined_history)

# SECTION 7: OPTIMIZED TRAINING EXECUTION

def train_model_optimized(model, train_gen, val_gen):
    """Execute optimized training process"""

    try:
        print("\n" + "="*80)
        print("🚀 STARTING OPTIMIZED MODEL TRAINING")
        print("="*80)
        print(f"📊 Training with {train_gen.samples:,} samples")
        print(f"📊 Validating with {val_gen.samples:,} samples")
        print(f"🌱 Organic samples: {organic_count:,}")
        print(f"🏭 Inorganic samples: {inorganic_count:,}")

        # Execute two-stage training
        history = fine_tune_model(model, train_gen, val_gen, initial_epochs=25)

        print("✅ Optimized model training completed successfully!")
        return history

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

# Execute optimized training
history = train_model_optimized(model, train_generator, validation_generator)


# SECTION 8: COMPREHENSIVE EVALUATION FOR LARGE DATASETS

def comprehensive_evaluation_large_dataset(model, val_generator):
    """Comprehensive evaluation optimized for large datasets"""

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)

    try:
        # Basic evaluation
        print("Computing validation metrics...")
        val_results = model.evaluate(val_generator, verbose=1)
        val_loss, val_accuracy = val_results[0], val_results[1]

        if len(val_results) > 2:
            val_precision, val_recall = val_results[2], val_results[3]
            f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
        else:
            val_precision = val_recall = f1_score = 0

        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Validation Loss: {val_loss:.4f}")
        if val_precision > 0:
            print(f"Validation Precision: {val_precision:.4f}")
            print(f"Validation Recall: {val_recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")

        # Sample-based evaluation for large datasets
        print("\nGenerating predictions on validation set...")
        val_generator.reset()

        # Get predictions in batches to handle large datasets
        all_predictions = []
        all_true_labels = []
        batch_count = 0
        max_batches = min(100, val_generator.samples // BATCH_SIZE)  # Limit for memory efficiency

        for batch_x, batch_y in val_generator:
            if batch_count >= max_batches:
                break

            pred_batch = model.predict(batch_x, verbose=0)
            all_predictions.extend(pred_batch.flatten())
            all_true_labels.extend(batch_y)
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"  Processed {batch_count}/{max_batches} batches...")

        y_pred = np.array(all_predictions)
        y_true = np.array(all_true_labels)
        y_pred_classes = (y_pred > 0.5).astype(int)

        # Detailed analysis
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred_classes)

        print(f"\nEvaluation sample size: {len(y_true):,} images")
        print(f"True classes present: {unique_true}")
        print(f"Predicted classes: {unique_pred}")

        # Class distribution
        print(f"\nTrue class distribution:")
        for cls in unique_true:
            count = np.sum(y_true == cls)
            percentage = count / len(y_true) * 100
            class_name = "Inorganic" if cls == 0 else "Organic"
            print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")

        print(f"\nPredicted class distribution:")
        for cls in unique_pred:
            count = np.sum(y_pred_classes == cls)
            percentage = count / len(y_pred_classes) * 100
            class_name = "Inorganic" if cls == 0 else "Organic"
            print(f"  {class_name}: {count:,} samples ({percentage:.1f}%)")

        # Confusion Matrix and Classification Report
        if len(unique_true) >= 2:
            cm = confusion_matrix(y_true, y_pred_classes)
            print(f"\nConfusion Matrix:")
            print(f"     Predicted")
            print(f"       0      1")
            print(f"True 0 {cm[0,0]:6d} {cm[0,1]:6d}")
            print(f"     1 {cm[1,0]:6d} {cm[1,1]:6d}")

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            labels = ['Inorganic', 'Organic']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Waste Classification - Confusion Matrix\n({len(y_true):,} samples)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()

            # Classification report
            print(f"\nClassification Report:")
            print("="*40)
            print(classification_report(y_true, y_pred_classes,
                                      target_names=['Inorganic', 'Organic'],
                                      labels=[0, 1],
                                      zero_division=0))

        # Performance assessment
        if val_accuracy >= 0.98:
            performance_level = "EXCEPTIONAL"
        elif val_accuracy >= 0.95:
            performance_level = "OUTSTANDING"
        elif val_accuracy >= 0.90:
            performance_level = "EXCELLENT"
        elif val_accuracy >= 0.85:
            performance_level = "GOOD"
        else:
            performance_level = "NEEDS IMPROVEMENT"

        print(f"\n{performance_level}: Model achieved {val_accuracy*100:.2f}% accuracy")
        print(f"Evaluated on {len(y_true):,} samples from {val_generator.samples:,} total validation images")

        # Data utilization summary
        total_samples = organic_count + inorganic_count
        print(f"\nDATA UTILIZATION SUMMARY:")
        print(f"Total training samples: {total_samples:,}")
        print(f"Organic samples: {organic_count:,} ({organic_count/total_samples*100:.1f}%)")
        print(f"Inorganic samples: {inorganic_count:,} ({inorganic_count/total_samples*100:.1f}%)")
        print(f"Class balance ratio: {min(organic_count, inorganic_count) / max(organic_count, inorganic_count):.3f}")

        return val_accuracy, y_pred, y_true

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 0.0, None, None

# Perform comprehensive evaluation
if history is not None:
    final_accuracy, predictions, true_labels = comprehensive_evaluation_large_dataset(model, validation_generator)
else:
    print("Skipping evaluation due to training failure")
    final_accuracy = 0.0

# SECTION 9: ADVANCED TRAINING VISUALIZATION

def plot_advanced_training_results(hist):
    """Plot comprehensive training results for large dataset training"""

    if hist is None:
        print("No training history to plot")
        return

    try:
        # Extract metrics
        metrics = ['accuracy', 'loss']
        if 'precision' in hist.history:
            metrics.extend(['precision', 'recall'])

        epochs = range(1, len(hist.history['accuracy']) + 1)

        # Create subplots
        fig_height = 6 if len(metrics) == 2 else 12
        plt.figure(figsize=(15, fig_height))

        # Plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(epochs, hist.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, hist.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        plt.title('Model Accuracy - Large Dataset Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, hist.history['loss'], 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, hist.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        plt.title('Model Loss - Large Dataset Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot precision if available
        if 'precision' in hist.history:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, hist.history['precision'], 'b-', label='Training Precision', linewidth=2)
            plt.plot(epochs, hist.history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
            plt.title('Model Precision - Large Dataset Training')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot recall
            plt.subplot(2, 2, 4)
            plt.plot(epochs, hist.history['recall'], 'b-', label='Training Recall', linewidth=2)
            plt.plot(epochs, hist.history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
            plt.title('Model Recall - Large Dataset Training')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        try:
            plt.savefig('/content/drive/MyDrive/optimized_training_results.png', dpi=300, bbox_inches='tight')
            print("📊 Training plots saved to Google Drive")
        except:
            try:
                plt.savefig('/content/optimized_training_results.png', dpi=300, bbox_inches='tight')
                print("📊 Training plots saved locally")
            except:
                print("Could not save training plots")

        plt.show()

        # Print training summary
        final_train_acc = hist.history['accuracy'][-1]
        final_val_acc = hist.history['val_accuracy'][-1]
        best_val_acc = max(hist.history['val_accuracy'])
        best_epoch = hist.history['val_accuracy'].index(best_val_acc) + 1

        print(f"\n📊 TRAINING SUMMARY:")
        print(f"├── Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
        print(f"├── Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
        print(f"├── Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"├── Best epoch: {best_epoch}")
        print(f"├── Total epochs trained: {len(epochs)}")
        print(f"└── Overfitting check: {'✅ Good' if abs(final_train_acc - final_val_acc) < 0.05 else '⚠️ Possible overfitting'}")

    except Exception as e:
        print(f"Error plotting training results: {e}")

# Plot training results
plot_advanced_training_results(history)

# SECTION 10: OPTIMIZED MODEL SAVING AND DEPLOYMENT

def save_optimized_model_complete(model):
    """Save optimized model with complete deployment package"""

    try:
        # Main save location
        model_dir = '/content/drive/MyDrive/optimized_waste_classifier_complete'
        os.makedirs(model_dir, exist_ok=True)

        # Save main model
        model_path = os.path.join(model_dir, 'optimized_waste_classifier.keras')
        model.save(model_path)

        # Save model weights separately for flexibility
        weights_path = os.path.join(model_dir, 'model_weights.h5')
        model.save_weights(weights_path)

        # Save model architecture
        architecture_path = os.path.join(model_dir, 'model_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(model.to_json())

        # Create comprehensive model info
        info_path = os.path.join(model_dir, 'model_info_complete.txt')
        with open(info_path, 'w') as f:
            f.write("OPTIMIZED WASTE CLASSIFICATION MODEL - LARGE DATASET VERSION\n")
            f.write("="*70 + "\n\n")
            f.write("MODEL SPECIFICATIONS:\n")
            f.write(f"• Architecture: EfficientNetB0 + Enhanced Classification Head\n")
            f.write(f"• Input Size: {IMG_SIZE}\n")
            f.write(f"• Total Parameters: {model.count_params():,}\n")
            f.write(f"• Batch Size: {BATCH_SIZE}\n")
            f.write(f"• Training Method: Two-stage (frozen + fine-tuned)\n\n")

            f.write("DATASET INFORMATION:\n")
            f.write(f"• Total Training Samples: {organic_count + inorganic_count:,}\n")
            f.write(f"• Organic Samples: {organic_count:,}\n")
            f.write(f"• Inorganic Samples: {inorganic_count:,}\n")
            f.write(f"• Class Balance Ratio: {min(organic_count, inorganic_count) / max(organic_count, inorganic_count):.3f}\n")
            f.write(f"• Data Sources: Multiple Kaggle datasets with intelligent sampling\n\n")

            f.write("PERFORMANCE:\n")
            f.write(f"• Final Validation Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
            f.write(f"• Model Status: {'✅ Deployment Ready' if final_accuracy > 0.90 else '⚠️ Needs Review'}\n\n")

            f.write("CLASSIFICATION:\n")
            f.write("• Class 0 (Inorganic): Non-biodegradable materials\n")
            f.write("  - Plastics, metals, glass, synthetic textiles, batteries\n")
            f.write("• Class 1 (Organic): Biodegradable materials\n")
            f.write("  - Fruits, vegetables, biological waste, paper, cardboard\n\n")

            f.write("FILES:\n")
            f.write("• optimized_waste_classifier.keras - Complete model\n")
            f.write("• model_weights.h5 - Model weights only\n")
            f.write("• model_architecture.json - Model architecture\n")
            f.write("• deployment_code.py - Ready-to-use inference code\n")

        # Create deployment code
        deployment_code_path = os.path.join(model_dir, 'deployment_code.py')
        with open(deployment_code_path, 'w') as f:
            f.write('''# Waste Classification - Deployment Code
import tensorflow as tf
import numpy as np
from PIL import Image

class WasteClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)

    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            img = Image.open(image_input)
        else:
            img = image_input
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0).astype(np.float32)

    def classify(self, image_input):
        try:
            img_array = self.preprocess_image(image_input)
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            predicted_class = int(prediction > 0.5)
            confidence = float(prediction if predicted_class == 1 else 1 - prediction)
            
            return {
                "class": predicted_class,
                "class_name": "Organic" if predicted_class == 1 else "Inorganic",
                "confidence": confidence
            }
        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

# Usage: classifier = WasteClassifier('model.keras')
# result = classifier.classify('image.jpg')
''')

        print(f"✅ Complete model package saved to: {model_dir}")
        print(f"📁 Files created:")
        print(f"  ├── optimized_waste_classifier.keras")
        print(f"  ├── model_weights.h5")
        print(f"  ├── model_architecture.json")
        print(f"  ├── model_info_complete.txt")
        print(f"  └── deployment_code.py")

        return model_path

    except Exception as e:
        print(f"Failed to save to Google Drive: {e}")

        try:
            # Save locally as backup
            model_dir = '/content/optimized_waste_classifier_backup'
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'optimized_waste_classifier.keras')
            model.save(model_path)
            print(f"✅ Model saved locally as backup: {model_path}")
            return model_path

        except Exception as e2:
            print(f"❌ Failed to save model: {e2}")
            return None

# Save the optimized model
if model is not None:
    saved_model_path = save_optimized_model_complete(model)
else:
    saved_model_path = None

# SECTION 11: INFERENCE FUNCTION

def classify_waste_optimized(url, model=None):
    """Classify waste image as organic or inorganic"""
    if model is None:
        return {"error": "No model available"}

    try:
        # Download and process image
        response = requests.get(url, timeout=15, stream=True)
        response.raise_for_status()
        
        img = Image.open(response.raw)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array.astype(np.float32), verbose=0)[0][0]
        predicted_class = int(prediction > 0.5)
        confidence = float(prediction if predicted_class == 1 else 1 - prediction)
        
        return {
            "class": predicted_class,
            "class_name": "Organic" if predicted_class == 1 else "Inorganic",
            "confidence": confidence,
            "raw_prediction": float(prediction)
        }
        
    except Exception as e:
        return {"error": f"Classification failed: {e}"}

# SECTION 12: FINAL SUMMARY

def generate_final_summary():
    """Generate concise summary of training process"""
    total_samples = organic_count + inorganic_count
    
    print("\n" + "="*60)
    print("WASTE CLASSIFICATION MODEL - TRAINING SUMMARY")
    print("="*60)
    
    print(f"Dataset: {total_samples:,} samples ({organic_count:,} organic, {inorganic_count:,} inorganic)")
    print(f"Architecture: EfficientNetB0 + Enhanced Classification Head")
    print(f"Training: Two-stage (frozen + fine-tuned)")
    print(f"Final Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Model Parameters: {model.count_params():,}")
    print(f"Model Saved: {'Yes' if saved_model_path else 'No'}")
    print(f"Training Status: {'Success' if history is not None else 'Failed'}")
    
    # Deployment readiness
    deployment_ready = (saved_model_path and final_accuracy >= 0.80 and total_samples >= 1000 and history is not None)
    print(f"Deployment Ready: {'Yes' if deployment_ready else 'No'}")
    
    if deployment_ready:
        print("\nModel is ready for deployment!")
        print("Usage: classify_waste_optimized('IMAGE_URL', model)")
    else:
        print("\nModel needs attention before deployment.")
    
    return {
        'total_samples': total_samples,
        'organic_count': organic_count,
        'inorganic_count': inorganic_count,
        'final_accuracy': final_accuracy,
        'model_saved': saved_model_path is not None,
        'training_successful': history is not None,
        'deployment_ready': deployment_ready
    }

# Generate final summary
final_summary = generate_final_summary()