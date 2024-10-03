# Date Fruit Classification using TensorFlow

This project demonstrates a neural network model for classifying various types of date fruit based on their attributes using supervised machine learning techniques. The dataset contains feature values representing the physical and chemical properties of each sample, and the `Class` column identifies the type of date fruit.

This README file provides a detailed explanation of the project steps, including pre-processing, model building, training, evaluation, and result visualization.

## Dataset Description

The dataset used in this project comes from an Excel file named `date_fruit.xlsx`. The dataset contains 34 features that describe various properties of the date fruits and a `Class` column, which indicates the fruit label (target variable).

The **shape of the dataset:** `(number_of_samples, number_of_features+target)`

- **Features (X):** 34 features representing various measurements related to date fruits.
- **Labels (y):** A categorical variable in the `Class` column representing different types of fruits.

## Prerequisites

Before running the code, ensure the following libraries are installed:

```bash
pip install pandas matplotlib scikit-learn tensorflow openpyxl
```

Here’s an overview of the main libraries used in this project:

- **Pandas** for data handling and manipulation.
- **Matplotlib** for plotting the training loss.
- **scikit-learn** for data preprocessing and splitting the dataset.
- **TensorFlow (Keras API)** for building and training the neural network.

## Project Structure

The project can be broken down into distinct sections:

### 1. Data Preprocessing

#### Steps:

1. **Importing Libraries and Reading the Dataset:**
   - Import necessary libraries such as `pandas`, `matplotlib`, `LabelEncoder` from `sklearn`, and `tensorflow`.
   - Load the dataset from `date_fruit.xlsx` into a Pandas DataFrame.

2. **Data Exploration & Preparation:**
   - Use `.shape` and `.unique()` methods to explore the dataset and identify the number of unique target classes.
   - Split the dataset into features (X) and target labels (y).
   - Normalize the features using scikit-learn’s `minmax_scale()` function.
   - Encode the categorical target labels into integer values using `LabelEncoder`.

3. **Splitting the Data:**
   - Use `train_test_split()` to split the dataset into training, validation, and test sets in a ratio of 80% training, 10% validation, and 10% testing.
   - The final structure includes:
     - 80% for `X_train`, `y_train`
     - 10% for `X_val`, `y_val`
     - 10% for `X_test`, `y_test`

### 2. Model Architecture

The neural network model consists of the following key layers:

1. **Input Layer:**
   - First layer has 4096 units and a `relu` activation function. Input shape is `(34,)` to match the feature size of the dataset.
   
2. **Hidden Layers:**
   - There are four hidden layers each containing 4096 units and using the `relu` activation function.
   - A dropout rate of `0.5` is added after each hidden layer to reduce overfitting by randomly turning off half of the neurons during training.

3. **Output Layer:**
   - The output layer consists of 7 units (representing 7 different classes of date fruits) with a `softmax` activation function for multi-class classification.

### 3. Model Training

The model is compiled using:

- **Loss Function:** Sparse categorical cross-entropy, which is suited for multi-class classification where the labels are encoded as integers.
- **Optimizer:** Adam optimizer for efficient convergence.
- **Metrics:** Accuracy is used to evaluate performance during training.

The model is trained for **100 epochs** with a batch size of **32**. Validation data is used to check the model's performance at the end of each epoch during training.

### 4. Visualization

After training the model, the training loss and validation loss are plotted using **Matplotlib**. This allows us to visually inspect how well the model is learning and if it's overfitting by comparing the loss on the training and validation sets.

### 5. Model Evaluation

The trained model is evaluated on the **test dataset** to assess its final performance. The `test_on_batch()` function is used to test the model on the batch of testing data, providing a summary of test accuracy and loss.

### 6. Results

- The final results provide an evaluation of test accuracy and loss.
- The training and validation loss plots are generated for visual inspection.

## Usage

To use this project, the following steps must be followed:

1. **Clone the repository (Optional):**
   
   This script can be used in a Jupyter Notebook or local Python environment.
2. **Install dependencies:**
   
   Install all necessary libraries listed in the *prerequisites* section.
   
3. **Load the dataset:**
   
   Ensure the file `date_fruit.xlsx` is available in the same directory.
   
4. **Run the script:**
   
   Execute the entire script in a Python environment or notebook for it to preprocess the data, build the model, train, evaluate, and visualize the outcome.
   
5. **View results:**
   
   Once the script runs, you’ll get:
   
   - A plot displaying how the model's loss evolves during training.
   - Final test results printed to the console.

## File Structure

```
.
├── date_fruit.xlsx  # Dataset with features and classes for different types of date fruit
├── date_fruit_classification.py  # Python file containing the data preprocessing, model training, and evaluation
└── README.md  # This README file
```
