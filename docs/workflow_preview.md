# Project Workflow Preview

## Project at a glance

**Task:** classify images into 37 cat and dog breeds  
**Dataset:** Oxford-IIIT Pet  
**Approach:** transfer learning with ImageNet-pretrained VGG16 and DenseNet121

## Workflow

```text
Oxford-IIIT Pet
      ↓
Check labels and train/validation/test splits
      ↓
Resize, augment training images, and normalize
      ↓
Load pretrained VGG16 or DenseNet121
      ↓
Replace the final classifier with a 37-class layer
      ↓
Train with CrossEntropyLoss and Adam
      ↓
Select the best checkpoint using validation loss
      ↓
Evaluate the selected configuration on the test set
      ↓
Save metrics, checkpoints, and plots
```

## Data

| Split | Images | Purpose |
| --- | ---: | --- |
| Train | 2,941 | Update model parameters |
| Validation | 739 | Track training and select checkpoints/configurations |
| Test | 3,669 | Final evaluation after selection |

Training images use random resized crops and horizontal flips. Validation and test images use fixed resizing and center cropping. All images are normalized with ImageNet statistics. The test split is kept separate from model selection.

## Models and training algorithm

- **VGG16:** convolutional layers extract image features in sequence; the final classifier is adapted to 37 classes.
- **DenseNet121:** dense blocks pass features between layers; its final classifier is adapted to 37 classes.
- Both models start from ImageNet weights. By default, the pretrained feature extractor is frozen and the new classifier is trained.
- The training loop uses **CrossEntropyLoss** and **Adam**. Validation loss is checked after each epoch; the best checkpoint is saved, and early stopping is used when validation loss stops improving.
- The recorded experiments compare learning rates `0.01` and `0.001` with batch size `16`. The best validation configuration for each architecture is evaluated on the test split.

## Repository structure

```text
BAI_E2/
├── bao_cao_hinh_anh/       # Figures used in reports
├── dataset/
│   ├── _oxford_raw/        # Original dataset files
│   ├── train/              # Training images, grouped by class
│   ├── val/                # Validation images, grouped by class
│   └── test/               # Test images, grouped by class
├── docs/                   # Reports, requirement notes, and this preview
├── notebooks/              # Jupyter report and dataset-checking script
├── outputs/                # Checkpoints, logs, and experiment results
├── src/
│   ├── data_loader.py      # Image transforms and DataLoaders
│   ├── engine.py           # Training and evaluation loops
│   ├── main.py             # Experiment setup and coordination
│   ├── models.py           # Shared model dispatcher
│   ├── vgg16_model.py      # VGG16 implementation
│   └── densenet121_model.py # DenseNet121 implementation
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and run instructions
```

## Suggested work split

Assign these work packages to available group members; combine roles if the group is small.

1. **Data:** inspect the dataset, verify class folders and splits, and document preprocessing.
2. **Models:** implement/adapt VGG16 and DenseNet121 and confirm the classifier outputs 37 classes.
3. **Training:** run the configured experiments, monitor validation metrics, and save checkpoints.
4. **Evaluation:** compare selected checkpoints on the test set and prepare result tables/figures.
5. **Report and integration:** check that code, metrics, figures, and explanations agree; prepare the notebook/report and presentation.

## Main project files

- `src/data_loader.py` — image preprocessing and data loading
- `src/models.py` — shared model selection and parameter counting
- `src/vgg16_model.py` — VGG16 model construction
- `src/densenet121_model.py` — DenseNet121 model construction
- `src/engine.py` — training and evaluation calculations
- `src/main.py` — experiment orchestration and result saving
- `notebooks/bao_cao_thuc_hanh_2_transfer_learning.ipynb` — detailed notebook report
