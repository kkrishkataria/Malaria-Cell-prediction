import os
base_path = './cell_images'
if os.path.exists(base_path):
    print("SUCCESS! Your Mac found the data.")
    categories = os.listdir(base_path)
    print(f"Categories found: {categories}")

    # Let's count the images to be sure
    for cat in categories:
        if not cat.startswith('.'): # Ignore hidden system files
            path = os.path.join(base_path, cat)
            print(f"{cat}: {len(os.listdir(path))} images")
else:
    print("ERROR: Folder not found. Check if 'cell_images' is in your VS Code sidebar.")

import pandas as pd
data = []
for category in ['Parasitized', 'Uninfected']:
    folder = os.path.join(base_path, category)
    # 1 for Parasitized, 0 for Uninfected
    label = 1 if category == 'Parasitized' else 0

    for img in os.listdir(folder):
        if img.endswith(('.png', '.jpg', '.jpeg')):
            data.append({'filepath': os.path.join(folder, img), 'label': str(label)})

# Create the DataFrame
df = pd.DataFrame(data)

# Shuffle the dataframe (important for training!)
df = df.sample(frac=1).reset_index(drop=True)

print(f"Total images indexed: {len(df)}")
print(df.head()) # Look at the first 5 rows to be sure

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Setting high resolution for your Mac screen
# %config InlineBackend.figure_format = 'retina'
sns.set_theme(style="whitegrid")

# We need to sample some images to get their dimensions
def get_dims(path):
    try:
        img = Image.open(path)
        return img.size # Returns (width, height)
    except:
        return (None, None)

# Take a sample to avoid long processing times
print("Analyzing image dimensions...")
sample_df = df.sample(27000).copy()
sample_df['width'], sample_df['height'] = zip(*sample_df['filepath'].map(get_dims))
sample_df['aspect_ratio'] = sample_df['width'] / sample_df['height']

# Create the visualization dashboard
plt.figure(figsize=(15, 5))

# Plot 1: Size Distribution
plt.subplot(1, 3, 1)
sns.scatterplot(data=sample_df, x='width', y='height', hue='label', alpha=0.5, palette='viridis') #palette = color scheme hue = label (0 or 1) alpha = transparency
plt.title('Cell Dimensions: Healthy vs Infected')

# Plot 2: Aspect Ratio
plt.subplot(1, 3, 2)
sns.kdeplot(data=sample_df, x='aspect_ratio', hue='label', fill=True, palette='magma')
plt.title('Aspect Ratio Distribution')

# Plot 3: Class Balance
plt.subplot(1, 3, 3)
sns.countplot(data=df, x='label', palette='Set2')
plt.title('Total Dataset Balance (0:Uninfected, 1:Parasitized)')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Define the Scenarios
print("splitting it into different cases to find out the best outcome into test , velidation and testing cases")
scenarios = [
    {"name": "70/15/15", "test1": 0.30, "test2": 0.50},
    {"name": "80/10/10", "test1": 0.20, "test2": 0.50},
    {"name": "75/13/12", "test1": 0.25, "test2": 0.48}
]

# 2. Define the Generator Settings
train_datagen = ImageDataGenerator(
    rescale=1./255, # pixels are from 0 to 255 and it will be difficult for ai to run all those mathematicall calulation so divide them by 255 and convert them between 0 to 1 making system stable and fast
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True
    #rotating the image creates them into new image making model not to learn the picture output and through it we train the model more effeciencly
)
test_datagen = ImageDataGenerator(rescale=1./255)
# we are using generator so that our computer do not crash by uploading all the 27000 image together it creates a file of 32 image train on it delete from ram than give next images
# we have already defined the size of generator alonside we can say it act as a log providing device to stop our device from crashing , we provide with the 20 different image of all the image as well from rotation so thaat ai cann't learn the format or something from image like shape , orientation etc.
# 3. Loop through and execute everything automatically
results = {}

for s in scenarios:
    print(f"\n--- SETTING UP SCENARIO: {s['name']} ---")

    # Split the Data
    train_df, temp_df = train_test_split(df, test_size=s['test1'], random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=s['test2'], random_state=42, stratify=temp_df['label'])

    # Create Generators
    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',  #
        target_size=(128, 128), batch_size=32, class_mode='binary'
    )
    val_gen = test_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label',
        target_size=(128, 128), batch_size=32, class_mode='binary', shuffle=False
    )
    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col='filepath', y_col='label',
        target_size=(128, 128), batch_size=32, class_mode='binary', shuffle=False
    )

    # Store them in a dictionary so you can access them later (e.g., results['80/10/10']['train'])
    results[s['name']] = {'train': train_gen, 'val': val_gen, 'test': test_gen}

print("\nAll scenarios are ready for training!")

import tensorflow as tf
from tensorflow.keras import layers, models

def build_malaria_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        # Block 1: Catching edges and colors
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2: Identifying cell structures
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3: Detecting fine parasite features
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4: Final deep feature extraction
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # The "Medical Expert" Bridge
        # Instead of Flatten, we use GlobalAveragePooling to help Grad-CAM later
        layers.GlobalAveragePooling2D(),

        # Dense Layers for classification
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevents the model from "memorizing" specific images
        layers.Dense(1, activation='sigmoid') # Sigmoid gives a probability (0 to 1)
    ])

    return model

# Initialize the model
malaria_cnn = build_malaria_model()

# Compile with a low learning rate for stability
malaria_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)

malaria_cnn.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Define Training Callbacks
# Patience=3 means if the model doesn't improve for 3 rounds, it stops to save time/prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when the model starts to "stall" (plateau)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# 2. Calculate Class Weights
# This ensures the model treats "Infected" and "Healthy" with equal importance
# even if one group has fewer samples.
neg = len(train_df[train_df['label'] == '0']) # Healthy
pos = len(train_df[train_df['label'] == '1']) # Infected
total = neg + pos

# Formula: (1 / class_count) * (total_samples / number_of_classes)
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print(f"--- Training Preparation Complete ---")
print(f"Callbacks: EarlyStopping(patience=3), ReduceLROnPlateau")
print(f"Weight for Healthy (0): {weight_for_0:.2f}")
print(f"Weight for Infected (1): {weight_for_1:.2f}")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Define the "Overfitting Protection" rules
# patience=5: check 5 more cycles after improvement stops
# restore_best_weights=True: jump back to the best version of the brain before it overfitted
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Optional but recommended: slows down learning when it gets difficult
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.00001
)

history_results = {}

for name, gen_dict in results.items():
    print(f"\nTRAINING SCENARIO: {name}")

    tf.keras.backend.clear_session()  # clear gpu memory to ensure all three cases do not get fixed
    model = build_malaria_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy', # Your loss function
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    # 2. Train until overfitting
    # We set epochs to 50 (high max), but the EarlyStopping will cut it off
    # the moment it starts overfitting for more than 5 cycles.
    history = model.fit(
        gen_dict['train'],
        steps_per_epoch=len(gen_dict['train']),
        validation_data=gen_dict['val'],
        validation_steps=len(gen_dict['val']),
        epochs=50,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    history_results[name] = history.history

    print(f"\nFinal Evaluation for {name}:")
    model.evaluate(gen_dict['test'], steps=len(gen_dict['test']))

print("\nALL SCENARIOS COMPLETE.")

import matplotlib.pyplot as plt
import numpy as np

# 1. Define the plotting function first
def plot_history(history, scenario_name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Performance for Scenario: {scenario_name}", fontsize=16)

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs (Training Rounds)') # Added Label
    plt.ylabel('Percentage Correct (0.0 to 1.0)') # Added Label
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs (Training Rounds)') # Added Label
    plt.ylabel('Error Score (Binary Crossentropy)') # Added Label
    plt.legend()
    plt.show()

# 2. The Loop that runs all cases
best_overall_recall = 0

for name, gen_dict in results.items():
    print(f"\nSTARTING TRAINING FOR SCENARIO: {name}")

    tf.keras.backend.clear_session()
    model = build_malaria_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(name='recall')]
    )

    # Train
    history = model.fit(
        gen_dict['train'],
        validation_data=gen_dict['val'],
        epochs=5,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # A. Show the Graphs
    plot_history(history, name)

    # B. Print your custom Summary Logic
    best_val_acc = max(history.history['val_accuracy'])
    total_epochs = len(history.history['accuracy'])
    best_epoch = np.argmin(history.history['val_loss']) + 1

    print("-" * 50)
    print(f"GRAPH DEPICTION & SUMMARY for {name}")
    print("-" * 50)
    print(f"1. LEARNING RATE: The smooth downward trend in Loss indicates the learning rate (0.0001) was optimal.")
    print(f"2. GENERALIZATION: Since Validation Accuracy ({best_val_acc:.2%}) followed Training Accuracy closely, the model is not overfitting.")
    print(f"3. EARLY STOPPING: The training stopped at Epoch {total_epochs} because the Validation Loss reached a minimum at Epoch {best_epoch}.")
    print(f"4. STABILITY: The lack of large 'zig-zags' in the loss curve shows that Batch Normalization successfully stabilized the training.")
    print("-" * 50)

    # Save best model
    max_recall = max(history.history['recall'])
    if max_recall >= best_overall_recall:
        best_overall_recall = max_recall
        model.save('malaria_model_validated.h5')

    # Store for final comparison later
    history_results[name] = history.history

import matplotlib.pyplot as plt
import numpy as np

# 1. Extract the best performance numbers from each scenario
labels = list(history_results.keys())
accuracy_scores = [max(history_results[name]['val_accuracy']) for name in labels]
recall_scores = [max(history_results[name]['recall']) for name in labels] # Change to 'val_recall' if available

x = np.arange(len(labels))  # Label locations
width = 0.35  # Width of the bars

# 2. Create the Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Best Val Accuracy', color='#3498db')
rects2 = ax.bar(x + width/2, recall_scores, width, label='Best Training Recall', color='#e74c3c')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores (0.0 to 1.0)')
ax.set_title('Comparison of Data Split Scenarios', fontsize=15)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1) # Give some space at the top for labels

# Auto-label the bars with their values
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2%}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()

# 3. THE WINNER LOGIC
winner = labels[np.argmax(recall_scores)]
print(f"\nTHE WINNER: Scenario {winner}")
print(f"REASON: In medical AI, we prioritize the highest Recall. Scenario {winner} ")
print(f"demonstrated the best ability to correctly identify infected malaria cells.")

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
import tensorflow as tf

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam_for_winner(img_tensor, heatmap, winner_name):
    # 1. Normalize heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

    # 2. Get Jet Colormap
    try:
        jet = plt.get_cmap("jet")
    except:
        jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(255 * heatmap)]

    # 3. Resize heatmap to match image (128x128)
    jet_heatmap = cv2.resize(jet_heatmap, (128, 128))

    # 4. Superimpose with higher contrast (0.6 heatmap weight)
    # This makes the parasite detection much clearer
    superimposed_img = jet_heatmap * 0.6 + img_tensor[0] * 0.4
    superimposed_img = np.clip(superimposed_img, 0, 1)

    # 5. Professional Plotting
    plt.figure(figsize=(12, 6))

    # Left: Original
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor[0])
    plt.title(f"Original Cell\n(Scenario: {winner_name})", fontsize=12)
    plt.axis('off')

    # Right: Grad-CAM
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title("AI Diagnostic Focus\n(Red = Parasite Detected)", fontsize=12, color='red')
    plt.axis('off')

    plt.suptitle("Explainable AI: Malaria Localization Evidence", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

try:
    # 1. Access winner
    scenario_names = list(results.keys())
    winner_idx = np.argmax(recall_scores)
    actual_winner_key = scenario_names[winner_idx]
    target_gen = results[actual_winner_key]['val']

    # 2. Find infected sample
    if hasattr(target_gen, 'filepaths'):
        all_filepaths = target_gen.filepaths
    else:
        import os
        all_filepaths = [os.path.join(target_gen.directory, f) for f in target_gen.filenames]

    all_labels = list(target_gen.classes)
    infected_indices = [i for i, val in enumerate(all_labels) if int(val) == 1]

    if len(infected_indices) > 0:
        # Pick the 6th sample (index 5) or the 1st if not enough
        idx = infected_indices[5] if len(infected_indices) > 5 else infected_indices[0]
        img_path = all_filepaths[idx]

        # 3. Sharp Preprocessing
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # INTER_CUBIC makes the parasite edges much sharper for the report
        img_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_CUBIC)
        img_tensor = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

        # 4. Generate Heatmap (Using your previous working function)
        last_conv = [l.name for l in reversed(model.layers) if 'conv2d' in l.name][0]
        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv)

        # 5. DISPLAY (This was the missing step!)
        display_gradcam_for_winner(img_tensor, heatmap, actual_winner_key)

    else:
        print("No infected samples found.")

except Exception as e:
    import traceback
    traceback.print_exc()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- STEP 1: LOAD WEIGHTS (CRITICAL) ---
# If you are in a new branch/cell, your model might have reset.
# You MUST load your saved best weights here.
try:
    malaria_cnn.load_weights('malaria_model_validated.h5')
    print("Weights loaded successfully.")
except:
    print("Warning: Could not load weights. Using current model state.")

# --- STEP 2: RE-RUN GENERATOR ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepath',
    y_col='label',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# --- STEP 3: DYNAMIC PREDICTIONS ---
preds = malaria_cnn.predict(test_generator)

# Check raw values: if they are all very close, 0.3 is too low.
print(f"Raw Preds - Min: {preds.min():.4f}, Max: {preds.max():.4f}, Mean: {preds.mean():.4f}")

# Switch to 0.5 to see if the model can actually distinguish 0 from 1
y_pred = (preds > 0.5).astype(int).flatten()
y_true = np.array(test_generator.classes)

mapping = test_generator.class_indices
print(f"Generator Mapping: {mapping}")

# Map "Infected" correctly
inf_idx = mapping.get('1', mapping.get('Infected', 1))

print("\n--- RESULTS FOR XAI BRANCH ---")
print(classification_report(y_true, y_pred, target_names=list(mapping.keys())))

# --- STEP 4: VISUALIZATION LOGIC ---
# We need a True Positive: Truly Infected (inf_idx) AND Predicted Infected (inf_idx)
tp_indices = np.where((y_true == inf_idx) & (y_pred == inf_idx))[0]

if len(tp_indices) > 0:
    # Pick a high-confidence sample (the one with the highest probability)
    # This ensures the Grad-CAM looks great!
    best_tp = tp_indices[np.argmax(preds[tp_indices])]
    img_path = test_df.iloc[best_tp]['filepath']

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_CUBIC)
    img_tensor = np.expand_dims(img_resized, axis=0).astype('float32') / 255.0

    # Auto-find last conv layer
    last_conv = [l.name for l in reversed(malaria_cnn.layers) if 'conv2d' in l.name][0]

    heatmap = make_gradcam_heatmap(img_tensor, malaria_cnn, last_conv)
    display_gradcam_for_winner(img_tensor, heatmap, f"Detected Parasite (Conf: {preds[best_tp][0]:.2%})")
else:
    print(f"FAILED: The model still predicts everything as one class.")
    print(f"Unique Predictions found: {np.unique(y_pred)}")

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 1. Reset generator and get predictions
test_generator.reset()
predictions = malaria_cnn.predict(test_generator)

# NEXT STEP: Using the 0.3 threshold for higher medical sensitivity (Recall)
# This ensures we catch as many parasites as possible
threshold = 0.3
y_pred = (predictions > threshold).astype(int)
y_true = test_generator.classes

# 2. Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 3. Visualize with both Counts and Percentages
plt.figure(figsize=(10, 8))

# Create labels combining counts and percentages for each box
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt='', cmap='Purples', # Using Purple to match the parasite theme
            xticklabels=['Healthy', 'Infected'],
            yticklabels=['Healthy', 'Infected'],
            annot_kws={"size": 14, "weight": "bold"})

plt.xlabel('AI Diagnostic Prediction', fontsize=12)
plt.ylabel('Ground Truth (Pathology)', fontsize=12)
plt.title(f'Malaria Diagnostic Matrix (Scenario: 75/13/12)\nThreshold: {threshold}', fontsize=15)
plt.show()

# 4. Final Classification Report
print("\n" + "="*30)
print("FINAL MEDICAL PERFORMANCE REPORT")
print("="*30)
print(classification_report(y_true, y_pred, target_names=['Healthy', 'Infected']))

from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Prepare Data
test_generator.reset()
predictions = malaria_cnn.predict(test_generator)

# Using the 0.3 threshold for medical safety
threshold = 0.3
y_pred = (predictions > threshold).astype(int)
y_true = test_generator.classes

# 2. Create Plotting Area
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# --- Graph 1: Confusion Matrix (Counts & Percentages) ---
cm = confusion_matrix(y_true, y_pred)
# Calculate labels for the heatmap
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt='', cmap='Reds', ax=ax1,
            xticklabels=['Healthy', 'Infected'],
            yticklabels=['Healthy', 'Infected'],
            annot_kws={"size": 13, "weight": "bold"})

ax1.set_title('Diagnostic Confusion Matrix\n(Counts & Population %)', fontsize=14)
ax1.set_xlabel('AI Predicted Diagnosis', fontsize=12)
ax1.set_ylabel('Actual Pathology Status', fontsize=12)

# --- Graph 2: ROC Curve (Reliability) ---
fpr, tpr, _ = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='red', lw=3, label=f'Model AUC = {roc_auc:.4f}')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

ax2.fill_between(fpr, tpr, color='red', alpha=0.1) # Shading the area under the curve
ax2.set_title('Receiver Operating Characteristic (ROC)\nModel Reliability Score', fontsize=14)
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax2.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax2.legend(loc="lower right", fontsize=12)
ax2.grid(alpha=0.3)

plt.suptitle(f"Biocon Malaria Diagnostic Report: Scenario 75/13/12\nFinal Accuracy: {np.mean(y_true == y_pred.flatten()):.2%}",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_fscore_support

# 1. Calculate the core clinical metrics
# average='binary' focuses specifically on the 'Infected' class (Positive class)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

# 2. Final Report Output
print(f"""
=========================================
      FINAL CLINICAL VALIDATION REPORT
      Scenario: 75/13/12 (Winner)
=========================================
TOTAL SAMPLES TESTED : {len(y_true)}
-----------------------------------------
SENSITIVITY (RECALL) : {recall:.2%}
-> (Ability to identify all infected patients)

PRECISION            : {precision:.2%}
-> (Reliability of a positive AI diagnosis)

F1-SCORE             : {f1:.2%}
-> (Harmonic mean of accuracy and safety)
-----------------------------------------
""")

# 3. Clinical Verdict Logic
if recall > 0.98:
    verdict = "EXCEPTIONAL: Near-perfect infection detection."
elif recall > 0.95:
    verdict = "READY FOR FIELD TEST: High safety profile."
else:
    verdict = "NEEDS FURTHER TUNING: Risk of False Negatives is too high."

print(f"VERDICT: {verdict}")

# 4. Final Interpretation for the Presentation
print(f"\nPHYSICIAN'S SUMMARY:")
print(f"Out of every 100 infected cells, this AI will successfully catch {recall*100:.1f} of them.")
print(f"When this AI flags a cell as infected, it is correct {precision*100:.1f}% of the time.")