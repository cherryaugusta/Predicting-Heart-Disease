# Predicting Heart Disease: A Supervised Machine Learning Analysis
## Project Description
According to estimates by the World Health Organization (WHO), cardiovascular diseases (CVDs) cause 17.9 million deaths annually.

The project utilizes the [Kaggle dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and employs a K-Nearest Neighbors classifier to predict the likelihood of a new patient having heart disease. The dataset includes various features such as age, sex, chest pain type, and other medical data. Before building the predictive model, exploratory data analysis will be conducted.

## EDA: Descriptive Statistics
**Key Tools**: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
**General comprehension of the data's features**

This project will begin with an exploration of the dataset. The information collected about each patient is as follows:

1. `Age`: Age of the patient [years]
2. `Sex`: Sex of the patient [M: Male, F: Female]
3. `ChestPainType`: Chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. `RestingBP`: Resting blood pressure [mm Hg]
5. `Cholesterol`: Serum cholesterol [mg/dL]
6. `FastingBS`: Fasting blood sugar [1: if FastingBS > 120 mg/dL, 0: otherwise]
7. `RestingECG`: Resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. `MaxHR`: Maximum heart rate achieved [numeric value between 60 and 202]
9. `ExerciseAngina`: Exercise-induced angina [Y: Yes, N: No]
10. `Oldpeak`: ST depression induced by exercise relative to rest [numeric value measured in mm]
11. `ST_Slope`: The slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. `HeartDisease`: Output class [1: heart disease, 0: normal]

The dataset comprises both numerical and categorical features. It is intriguing to examine the data type of each column.

This project exploration focuses on the numerical variables first. The dataset has seven numerical features and five categorical features. It is important to note that two of the numerical features, "FastingBS" and "HeartDisease," are actually categorical.

**Observations**:
- The average age of the patients is approximately 53.5 years.
- There is an indication of a left-skewed distribution and the possibility of outliers skewing the distribution. This is reflected by the median "Cholesterol" being higher than the mean by roughly 25 mm/dl.
- The lowest value is zero for both "RestingBP" and "Cholesterol." According to the [American Heart Association](https://www.heart.org/en/health-topics/cholesterol/about-cholesterol/what-your-cholesterol-levels-mean), it is unlikely for "Cholesterol" to be 0. Therefore, it is essential to clean both of these values later. Serum cholesterol is a composite of different measurements, and it is impossible for "RestingBP" to be 0.
- There is no indication of any missing values in these columns, but this needs to be confirmed across the entire dataset.
- Regarding the categorical variables, it would also be relevant to explore their relationships with the target feature, "HeartDisease." Therefore, it is essential to check if there are any missing values in the dataset.
- There is no missing values in the dataset
- In term of identification of dtype object of the categorical columns, based on the number of unique values in each column, it can be verified that those columns are categorical. However, this information alone does not provide additional insights. Additionally, `FastingBS` and `HeartDisease` are also categorical variables because they consist solely of binary values, which we can easily confirm.
- The dataset exhibits a significant gender imbalance, with `725` male patients compared to `193` female patients. This imbalance may introduce bias into our model.
- There are `496` patients who experienced asymptotic (ASY) chest pain.
- A total of `552` patients had a normal resting ECG.
- `704` patients had a blood sugar level below `120` mg/dl.
Analyzing these variables in relation to `HeartDisease` will provide a clearer understanding of the data distribution.

**Findings**: The dataset shows a noticeable skew toward male patients, with only 50 female patients diagnosed with heart disease. A substantial portion of patients, 392 in total, diagnosed with heart disease experienced asymptomatic (ASY) chest pain. While chest pain might be a significant feature for our model, "asymptomatic" suggests these patients did not exhibit chest pain as a symptom.

Additionally, a considerable number of patients (170) with blood sugar levels above 120 mg/dl were diagnosed with heart disease compared to those who were not. Among patients who experienced exercise-induced angina, 316 were diagnosed with heart disease. Similarly, of those with a flat ST slope, 381 were diagnosed with heart disease.

Based on the data distribution from the plots above, it is relevant to identify potential features of interest. However, it is essential to first clean up the dataset before refining our feature selection.

## Data Cleaning
The initial analysis shows that there are no missing values in the dataset. However, certain columns contain zero values that appear inconsistent.
It is important to assess the number of zero values present in the `RestingBP` and `Cholesterol` columns and determine the appropriate approach for addressing these anomalies.
**Findings**: The variable `RestingBP` contains only a single zero value, so we can exclude that row from our analysis. On the other hand, there are `172` zero values for `Cholesterol`, which is a substantial number. Removing all these zero values is not feasible, and substituting them with the median might not be the most suitable solution, but we will proceed with this approach for now.
To enhance accuracy, we will replace the zero values in `Cholesterol` based on the presence or absence of heart disease. Specifically, for patients diagnosed with heart disease, we will substitute the zero values with the median of the non-zero `Cholesterol` values from other patients with heart disease. Similarly, for patients without heart disease, the zero values will be replaced with the median of the non-zero `Cholesterol` values from patients without heart disease.
The minimum values for both have been updated, and there are no longer any zero values present in either dataset.

## Feature Selection

Based on the earlier exploratory data analysis (EDA) and a general comprehension of the features, it is relevant to initially focus on the following attributes:

- Age
- Sex
- ChestPainType
- Cholesterol
- FastingBS

Additionally, we will assess the strength of the correlation between these feature columns and the target column to refine our selection.

To facilitate this process, we will first transform our categorical variables into dummy variables.

It is relevant to determine the extent of their correlation.

From the correlation heatmap, it is observed that the following features exhibit a positive correlation (correlation coefficient exceeding 0.3) with HeartDisease:

Oldpeak
MaxHR
ChestPainType_ATA
ExerciseAngina_Y
ST_Slope_Flat
ST_Slope_Up
The threshold for the correlation coefficient was selected somewhat arbitrarily. Notably, Cholesterol does not show a strong correlation with HeartDisease, so we might consider excluding it for now.

Based on the current findings, it is important to refine the feature set to:

Oldpeak
Sex_M (although it has a relatively low correlation coefficient, it was significant in our exploratory data analysis)
ExerciseAngina_Y
ST_Slope_Flat
ST_Slope_Up

Let's proceed with building model using these features

## Building a Classifier with a Single Feature
Initially, the dataset is divided into two subsets: a training set and a test set.
The process will start with creating a model for each of the features mentioned above and evaluating their performance using accuracy as the metric.

## Building a Classifier with Multiple Features
Before training the model using all the features mentioned, it's important to normalize the data by applying scikit-learn's MinMaxScaler to scale the values to a range between 0 and 1, and then proceed with training the model again.

## Hyperparameter Tuning
Best Score: 0.8498
Best Parameters: {'metric': 'minkowski', 'n_neighbors': 13}
Accuracy on the test set: 0.8043

## Observation Summary:
The final model was trained using the following features:

- `Oldpeak`
- `Sex_M`
- `ExerciseAngina_Y`
- `ST_Slope_Flat`
- `ST_Slodditional data.
- The accuracy is 0.8043. The model's performance is better on the test set compared to the  However, this accuracy may not fully reflect the model's performance due to the limitations of our dataset.
validation set.
- The test set accuracy is lower than that of the training set, which could be attributed to overfitting/generalization.
- Potential improvements to performance include feature engineering, more data, fine-tuning hyperparameters.
- Pros of using this model in a real-world healthcare setting include [mention strengths, e.g., interpretability, simplicity].
- Cons of using this model in a real-world healthcare setting include [mention weaknesses, e.g., sensitivity to imbalanced data, limited interpretability].


To potentially improve the results, several next steps can be considered:

- Experiment with different feature sets.
- Broaden the grid search to discover more optimal hyperparameters.
- Investigate other algorithms that could outperform k-NN.
- Attempt to gather additional data interpretability.
