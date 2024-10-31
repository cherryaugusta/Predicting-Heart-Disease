# Predicting Heart Disease: A Supervised Machine Learning Analysis

This project is a solution to a guided project from Dataquest, originally titled ["Predicting Heart Disease"](https://www.dataquest.io/projects/guided-project-a-predicting-heart-disease/). This project is shared for educational and portfolio purposes only to showcase my understanding and skills. I have completed this project by adding my insights, adjustments, and interpretations.

## Project Introduction
Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for an estimated 17.9 million deaths annually, according to the World Health Organization (WHO).

The project utilizes the [Kaggle dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) to build a predictive model for heart disease. A K-Nearest Neighbors (KNN) classifier was employed to predict the likelihood of heart disease in new patients. The dataset includes various features such as age, sex, chest pain type, and other medical attributes. Prior to model development, exploratory data analysis (EDA) was conducted to understand the dataset and identify key patterns.

**Key Tools**: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

## Exploratory Data Analysis (EDA): Descriptive Statistics
### Data Overview

The dataset contains several features, providing demographic and medical information on each patient, including:

1. `Age`: Age of the patient (years)
2. `Sex`: Sex of the patient (M: Male, F: Female)
3. `ChestPainType`: Type of chest pain (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)
4. `RestingBP`: Resting blood pressure (mm Hg)
5. `Cholesterol`: Serum cholesterol (mg/dL)
6. `FastingBS`: Fasting blood sugar (1: if FastingBS > 120 mg/dL, 0: otherwise)
7. `RestingECG`: Resting electrocardiogram results (Normal, ST: ST-T wave abnormality, LVH: Left Ventricular Hypertrophy)
8. `MaxHR`: Maximum heart rate achieved (between 60 and 202)
9. `ExerciseAngina`: Exercise-induced angina (Y: Yes, N: No)
10. `Oldpeak`: ST depression induced by exercise relative to rest (mm)
11. `ST_Slope`: The slope of the peak exercise ST segment (Up, Flat, Down)
12. `HeartDisease`: Target class (1: heart disease, 0: no heart disease)

### Numerical vs. Categorical Features

The dataset contains both numerical and categorical variables. Although some variables, such as `FastingBS` and `HeartDisease`, are numerical in nature, they are better categorized as binary features.

### Key Observations

- The mean age of the patients is approximately 53.5 years.
- A left-skewed distribution is evident in some variables, with possible outliers. For instance, the median cholesterol level is higher than the mean by around 25 mg/dL, suggesting skewness.
- Both `RestingBP` and `Cholesterol` have minimum values of zero, which are biologically implausible for these variables. These inconsistencies will require data cleaning.
- No missing values were detected in the dataset.
- The dataset reveals a significant gender imbalance, with 725 male patients and 193 female patients.
- Most patients (496) reported asymptomatic (ASY) chest pain, while 552 had normal resting ECG results, and 704 had fasting blood sugar levels below 120 mg/dL.

### Findings

The dataset is heavily skewed toward male patients, with only 50 female patients diagnosed with heart disease. Among the patients diagnosed with heart disease, 392 reported asymptomatic chest pain, suggesting that chest pain may not be a reliable indicator for predicting heart disease. Furthermore, blood sugar levels above 120 mg/dL and flat ST slopes appear to have a strong association with heart disease, as indicated by the high prevalence of heart disease in these groups.

These findings will inform the next steps in feature selection and model building. However, cleaning the dataset is necessary before refining the feature selection process.

## Data Cleaning

The dataset does not contain any missing values, but some variables, such as `RestingBP` and `Cholesterol`, contain zero values, which are biologically implausible. Specifically, `RestingBP` contains one zero value, which will be excluded, and `Cholesterol` has 172 zero values, which is a significant proportion. Removing all these zero values is not a feasible approach. Therefore, the approach taken was to impute these values by replacing them with the median cholesterol values, stratified by heart disease diagnosis.

### Findings

After imputation, the minimum values in both `RestingBP` and `Cholesterol` have been updated, and no zero values remain.

## Feature Selection

Based on the EDA, the following initial features were considered for model building:

- Age
- Sex
- ChestPainType
- Cholesterol
- FastingBS

To refine this selection, the correlation between these features and the target variable, `HeartDisease`, was examined. The correlation heatmap revealed that certain features, such as `Oldpeak`, `MaxHR`, `ChestPainType_ATA`, `ExerciseAngina_Y`, `ST_Slope_Flat`, and `ST_Slope_Up`, exhibited stronger correlations with `HeartDisease` (correlation coefficient > 0.3). Interestingly, `Cholesterol` did not show a strong correlation and was excluded from further analysis.

The final set of features selected for model training includes:

- `Oldpeak`
- `Sex_M`
- `ExerciseAngina_Y`
- `ST_Slope_Flat`
- `ST_Slope_Up`

## Model Development

### Single-Feature Classifier

Initially, models were trained using individual features to evaluate their predictive power. Accuracy was used as the evaluation metric.

### Multi-Feature Classifier

A model was then built using the selected features. To improve model performance, scikit-learn's MinMaxScaler was applied to normalize the data, scaling all features to a range between 0 and 1.

## Hyperparameter Tuning

Using grid search, the hyperparameters for the KNN classifier were optimized. The best results were achieved with the following configuration:

- Best Score: 0.8498
- Best Parameters: `{'metric': 'minkowski', 'n_neighbors': 13}`
- Test Set Accuracy: 0.8043

## Conclusion

The final model was trained using the following features:

- `Oldpeak`
- `Sex_M`
- `ExerciseAngina_Y`
- `ST_Slope_Flat`
- `ST_Slope_Up`

The test set accuracy was 0.8043, which, while reasonable, suggests room for improvement. The lower test accuracy compared to the training accuracy may indicate overfitting or generalization issues.

### Strengths and Weaknesses

Strengths of this model include its simplicity and interpretability, which are beneficial in a healthcare setting. However, the model is sensitive to imbalanced data, such as the gender disparity in the dataset. Additionally, the performance might be limited due to the relatively small feature set and the use of a single algorithm.

### Future Work

To further improve the model's performance, the following steps are recommended:

- Experiment with alternative feature sets to enhance predictive power.
- Broaden the grid search to explore a wider range of hyperparameters.
- Investigate alternative algorithms, such as decision trees or support vector machines, which may outperform KNN.
- Collect additional data to address issues of bias and improve model generalizability.
