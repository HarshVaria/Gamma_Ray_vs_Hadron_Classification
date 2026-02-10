# Gamma/Hadron Classifier

## Project Overview
This project aims to classify atmospheric showers as either gamma-ray or hadron-induced using various machine learning models. The dataset consists of features extracted from images of these showers, captured by a ground-based atmospheric Cherenkov telescope.

## Dataset
The dataset used is the "MAGIC Gamma Telescope Data Set" from the UCI Machine Learning Repository.

**Source:**
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

**Features:**
The dataset includes 10 features (fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist) describing the images and a 'class' label indicating 'g' (gamma) or 'h' (hadron).

## Data Preprocessing
1.  **Loading Data**: The `magic04.data` CSV file is loaded into a pandas DataFrame.
2.  **Target Encoding**: The 'class' column is converted from categorical ('g', 'h') to numerical (1 for gamma, 0 for hadron).
3.  **Data Splitting**: The dataset is split into training, validation, and test sets with a 60/20/20 ratio.
4.  **Feature Scaling**: All numerical features are scaled using `StandardScaler` to normalize their ranges.
5.  **Oversampling**: To address class imbalance, `RandomOverSampler` is applied to the training dataset.

## Models Implemented
Several classification models were implemented and evaluated:

### 1. k-Nearest Neighbors (kNN)
*   **Model**: `KNeighborsClassifier` with `n_neighbors=5`.
*   **Performance on Test Set:**
    ```
                   precision    recall  f1-score   support

            0       0.74      0.73      0.73      1305
            1       0.86      0.87      0.86      2499

     accuracy                           0.82      3804
    macro avg       0.80      0.80      0.80      3804
 weighted avg       0.82      0.82      0.82      3804
    ```

### 2. Naive Bayes (GaussianNB)
*   **Model**: `GaussianNB`.
*   **Performance on Test Set:**
    ```
                   precision    recall  f1-score   support

            0       0.63      0.43      0.51      1305
            1       0.75      0.87      0.80      2499

     accuracy                           0.72      3804
    macro avg       0.69      0.65      0.66      3804
 weighted avg       0.71      0.72      0.70      3804
    ```

### 3. Logistic Regression
*   **Model**: `LogisticRegression`.
*   **Performance on Test Set:**
    ```
                   precision    recall  f1-score   support

            0       0.65      0.71      0.68      1305
            1       0.84      0.80      0.82      2499

     accuracy                           0.77      3804
    macro avg       0.75      0.76      0.75      3804
 weighted avg       0.78      0.77      0.77      3804
    ```

### 4. Support Vector Machine (SVM)
*   **Model**: `SVC` (default parameters).
*   **Performance on Test Set:**
    ```
                   precision    recall  f1-score   support

            0       0.80      0.81      0.80      1305
            1       0.90      0.90      0.90      2499

     accuracy                           0.87      3804
    macro avg       0.85      0.85      0.85      3804
 weighted avg       0.87      0.87      0.87      3804
    ```

### 5. Neural Network (TensorFlow/Keras)
*   **Architecture**: A sequential model with two dense ReLU layers and a sigmoid output layer, including dropout for regularization.
*   **Hyperparameter Tuning**: A grid search was performed over `num_nodes` ([16, 32, 64]), `dropout_prob` ([0, 0.2]), `learning_rate` ([0.01, 0.005, 0.001]), and `batch_size` ([32, 64, 128]) to find the model with the least validation loss.
*   **Performance on Test Set (Best Model):**
    ```
                   precision    recall  f1-score   support

            0       0.85      0.77      0.81      1305
            1       0.89      0.93      0.91      2499

     accuracy                           0.87      3804
    macro avg       0.87      0.85      0.86      3804
 weighted avg       0.87      0.87      0.87      3804
    ```

## Conclusion
Both the Support Vector Machine (SVM) and the tuned Neural Network models achieved the highest accuracy and F1-scores, demonstrating strong performance in classifying gamma and hadron particles from the given features. The Neural Network, after hyperparameter tuning, showed comparable or slightly better overall performance, particularly in terms of macro average metrics, suggesting its robustness.
