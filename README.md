# Diabetes Prediction Model  

 files related to the Time Delivery Prediction project.

- **README.md:** This file serves as the main documentation for the project, providing details on the business case, methodology, and findings.

- **notebook.ipynb:** This Jupyter Notebook contains code for data exploration and presentation purposes.

- **requirement.txt:** This file lists the project requirements and dependencies needed to run the code.

## Installation

To install the required dependencies, run the following command:

```pip install -r requirements.txt```

The aim of the project is to construct a machine learning model that is able to predict whether an individual has diabetes or not based on certain health parameters. With the use of the Pima Indians Diabetes Database, the project aims to identify patterns and predict the likelihood of diabetes and thus help health professionals identify possibly at-risk individuals for early intervention.

**Pima Indians Diabetes Database** on Kaggle   
This dataset was initially provided by the National Institute of Diabetes and Digestive and Kidney Diseases. Its primary purpose is to aid in the diagnostic prediction of diabetes in patients, using various medical measurements included in the dataset. The data was curated with specific criteria: all individuals are female, at least 21 years old, and of Pima Indian descent.

Link to access the data: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### Features of data:  
- **Observations**: 6921 (769 instances x 9 features)  
- **Pregnancies**: The number of pregnancies the individual has had
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (kg/m²)
- **DiabetesPedigreeFunction**: A score that represents family history of diabetes
- **Age**: Age of the individual (years)
- **Outcome**: 1= has diabetes; 0= does not have diabetes  

### Introduction:  
Early diagnosis is very important in managing diabetes, a chronic disease that causes serious health complications. Diabetes prediction has been made possible with machine learning (ML), but creating precise models remains very challenging because of the nature of the disease and the large differentials between patient profiles. Most machine learning (ML) algorithms, including K-Nearest Neighbors (K-NN), Random Forest, and boosting algorithms, have been tried to enhance prediction. There have been many research papers praising these methods, with some algorithms performing better than others on some datasets or in some situations. There are different works with ML for diabetes detection that used the Pima Indian dataset. For example, in [1], The study utilized the Pima Indians Diabetes Dataset, consisting of 768 patients with 9 features, and experimented with data balancing techniques like SMOTE and dimensionality reduction using PCA. The best-performing models were K-NN and BNB, achieving accuracies of 79.6% and 77.2%, respectively, when trained on SMOTE-augmented data. However, performance significantly dropped when PCA was applied, indicating potential limitations in feature reduction for this task. The study relied on traditional evaluation metrics (accuracy, precision, recall, F1-score) and concluded that K-NN and BNB were the most effective models, though with only marginal differences in performan




### Collaborators:  
**Thao Nguyen** (coding), **Nhi Nguyen** (researching and writing information for readme file)  

### Conclusion: 
 
In this project, we implemented various machine learning techniques to analyze the dataset, optimize model performance, and identify meaningful clusters within the data. Our approach included hyperparameter optimization using Optuna, clustering with K-Means, and evaluating model performance through classification metrics and visualizations.

Model Performance Evaluation with Optuna
Using Optuna, we tuned hyperparameters for several classification models, including Random Forest, SVM, Logistic Regression, and XGBoost. Based on the evaluation metrics (Accuracy, Precision, Recall, F1-Score, and AUC), we found that XGBoost performed best for this dataset

Clustering with K-Means
We applied K-Means clustering and determined that three clusters (k=3) were the optimal choice, successfully grouping the data into meaningful segments. The characteristics of each cluster are as follows:

Cluster 0 (Low-Risk Group)

- Lowest Glucose (104.5) → Likely non-diabetic or low risk.
- Lower BMI (29.54) → Mostly in the normal/overweight range.
- Lowest Insulin (56.25) → Suggests better metabolic health.
- Very Low Diabetes Incidence (Outcome = 0.07) → Only 7% have diabetes.
- Younger Age Group → This cluster consists mostly of younger individuals.

Cluster 1 (High-Risk Group)

- Highest Glucose (149.5) → Strong indicator of diabetes.
- Highest BMI (37.59) → Falls in the obese range, increasing risk.
- Highest Insulin Levels (211.93) → Suggests insulin resistance.
- Very High Diabetes Incidence (Outcome = 0.75) → 75% of individuals have diabetes.
- Older Age Group → Aging population with higher risk factors.

Cluster 2 (Intermediate Group)

- Moderate Glucose (128.4) → Higher than Cluster 0 but lower than Cluster 1.
- Moderate BMI (32.14) → Falls into the overweight/obese category.
- Surprisingly Low Insulin (23.96) → Could indicate metabolic variations.
- Moderate Diabetes Incidence (Outcome = 0.54) → 54% of individuals have diabetes.
- Oldest Age Group → This cluster represents the oldest individuals.

=> Cluster 1 has the highest diabetes risk, suggesting that interventions should target individuals with high glucose, high BMI, and insulin resistance.
=> Cluster 0 appears to be the healthiest, with lower glucose, insulin, and BMI levels.

*This project demonstrates the power of machine learning, clustering, and hyperparameter tuning in extracting valuable insights from healthcare-related data. Future work could involve feature engineering, deep learning models, or alternative clustering techniques (DBSCAN) to further refine predictions and segmentations.

### Further Improvements:  

1. Expand the Dataset for Generalizability
The current dataset only includes 769 female individuals of Pima Indian heritage, which limits model generalization.
Specific ways to improve it:
- Find and merge additional diabetes-related datasets from different demographics (from the UCI Machine Learning Repository, we found one: https://archive.ics.uci.edu/dataset/34/diabetes).
- If possible, we will try out with a completely different dataset that will represent the US population: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- For the first case, we will use pandas to join datasets with consistent features.
  
2. Test More Machine Learning and Deep Learning Algorithms
Currently, we only limited to 4 models. We can improve the prediction results by either improving the data preprocessing step or by choosing completely different modelling methods.

For the former, we will spend time working again on:
- Handle Missing and Zero-Like Values More Effectively: In the current dataset, 0 values in these columns likely represent missing or incorrectly recorded data, not true physiological zeros. We may need to work on that more closely. If the 0 is not the true data, but actually to indicate that value is missing, we will need to change those values to null values using np.nan(), and apply imputation methods that are going to work best for the dataset (median, mean,...)
- Handle Outliers: In this project, we have used one method to handle outliers, but we believe there could be better ways to handle it. For future improvement, we will work on trying out different methods to effectively handle outliers or we possibly use log transform to transform variable to make it normalized.
- Do further Feature Engineering because original features may not capture complex interactions.: we will create new features such as: age-to-BMI ratio, Glucose-to-Insulin index, Pregnancy rate (pregnancies divided by age), Binned Age Groups (e.g., 21–30, 31–40, etc.).

For the latter, We will implement algorithms such as:
- SVM with RBF kernel (for nonlinear classification)
- Deep learning algorithms (lstm). This may need us to change the data because deep learning requires large number of observations 
we will also try out GridSearchCV or RandomizedSearchCV to optimize hyperparameters instead of limiting to only Optuna.

3. In addition, Our project currently lacks statistical testing to explore different feature relationships. For further analysis we will design Hypothesis Tests for Feature Relationships:
- We will work on testing hypotheses like: “Is high glucose always accompanied by high insulin levels?”; “Does age moderate the relationship between BMI and diabetes?”;
- We will Use statistical tests like Pearson correlation, ANOVA for comparing groups.

### References: 
1. Iparraguirre-Villanueva, O., Espinola-Linares, K., Flores Castañeda, R. O., & Cabanillas-Carbonell, M. (2023, July 15). Application of machine learning models for early detection and accurate classification of type 2 diabetes. Diagnostics (Basel, Switzerland). https://pmc.ncbi.nlm.nih.gov/articles/PMC10378239/#sec5-diagnostics-13-02383  
 
2. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An introduction to statistical learning: With applications in Python. Springer.  

3. Learning, U. M. (2016, October 6). Pima Indians Diabetes Database. Kaggle. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

4. Shams, M. Y., Tarek, Z., & Elshewey, A. M. (2025, January 6). A novel RFE-GRU model for diabetes classification using Pima Indian Dataset. Nature News. https://www.nature.com/articles/s41598-024-82420-9#:~:text=Introducing%20a%20variety%20of%20machine,Histogram%20Gradient%20Boosting%20(HGB). 
 

