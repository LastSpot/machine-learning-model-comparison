# Machine Learning Model Comparison Project

This project implements and compares three different machine learning algorithms (Neural Network, K-Nearest Neighbors, and Random Forest) on various datasets to evaluate their performance in classification tasks.

## Project Overview

The project includes custom implementations of:
- Neural Network with configurable layers and neurons
- K-Nearest Neighbors classifier
- Random Forest classifier

These algorithms are tested on four different datasets:
1. Digits Dataset (from sklearn)
2. Titanic Dataset
3. Loan Prediction Dataset
4. Parkinson's Disease Dataset

## Project Structure

```
.
├── data/                   # Contains the dataset files
├── figures/               # Generated plots and visualizations
├── michael_ML/           # Custom ML implementations
│   ├── neural_network.py
│   ├── k_NN.py
│   └── random_forest.py
├── parameters/           # Model parameters and configurations
├── tables/              # Generated result tables
├── main.py             # Main execution script
└── experiments.ipynb   # Jupyter notebook for additional experiments
```

## Features

- Custom implementation of three popular machine learning algorithms
- Automated parameter tuning using k-fold cross-validation
- Data preprocessing including normalization and one-hot encoding
- Support for both numerical and categorical features
- Performance evaluation metrics and visualizations
- Stratified k-fold cross-validation for robust model evaluation

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage

1. Ensure all required datasets are in the `data/` directory
2. Run the main script:
   ```bash
   python main.py
   ```
   This will:
   - Load and preprocess all datasets
   - Train all three models on each dataset
   - Generate performance metrics and visualizations
   - Save results in the `figures/` and `tables/` directories

3. For additional experiments and analysis, use the `experiments.ipynb` notebook

## Implementation Details

### Neural Network
- Configurable number of hidden layers and neurons
- Sigmoid activation function
- Gradient descent optimization
- Support for regularization
- Automated hyperparameter tuning

### K-Nearest Neighbors
- Custom distance metric implementation
- Support for both numerical and categorical features
- Automated k-value optimization

### Random Forest
- Ensemble learning method
- Decision tree implementation
- Feature importance analysis
- Random feature subset selection

## Results

The results for each model and dataset combination are stored in:
- `figures/`: Learning curves and performance plots
- `tables/`: Detailed metrics including accuracy, precision, recall, and F1-score

## License

This project is licensed under the terms included in the LICENSE file. 