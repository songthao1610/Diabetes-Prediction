# Diabetes Prediction Model  
The aim of the project is to construct a machine learning model that is able to predict whether an individual has diabetes or not based on certain health parameters. With the use of the Pima Indians Diabetes Database, the project aims to identify patterns and predict the likelihood of diabetes and thus help health professionals identify possibly at-risk individuals for early intervention.
### Dataset:   
**Pima Indians Diabetes Database** on Kaggle   
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
Early diagnosis is very important in managing diabetes, a chronic disease that causes serious health complications. Diabetes prediction has been made possible with machine learning (ML), but creating precise models remains very challenging because of the nature of the disease and the large differentials between patient profiles. Most machine learning (ML) algorithms, including K-Nearest Neighbors (K-NN), Random Forest, and boosting algorithms, have been tried to enhance prediction. There have been many research papers praising these methods, with some algorithms performing better than others on some datasets or in some situations. Model performance also relies heavily on feature selection and preprocessing, such as normalization of data and missing value handling.  

### Collaborators:  
**Thao Nguyen** (coding), **Nhi Nguyen** (researching and writing information for readme file)  

### Further Improvements:  

Without the time constraints of meeting the project deadline, we would focus on obtaining a larger and more diverse dataset. This model could then be further improved by testing additional machine-learning algorithms and new ways of transforming or merging the current data into more useful variables. For example, we can create a new feature like age-to-BMI ratio because these two variables usually relate. This may help the model capture more representative patterns and improve its prediction accuracy.  

### References: 
Chen, P., & Pan, C. (2018, March 27). Diabetes classification model based on boosting algorithms - BMC Bioinformatics. BioMed Central. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2090-9#:~:text=Non%2Dparametric%20statistical%20testing%20is,operating%20characteristic%20curve%20reached%200.99.  
 
Iparraguirre-Villanueva, O., Espinola-Linares, K., Flores Castañeda, R. O., & Cabanillas-Carbonell, M. (2023, July 15). Application of machine learning models for early detection and accurate classification of type 2 diabetes. Diagnostics (Basel, Switzerland). https://pmc.ncbi.nlm.nih.gov/articles/PMC10378239/#sec5-diagnostics-13-02383  
 
James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An introduction to statistical learning: With applications in Python. Springer.  

Learning, U. M. (2016, October 6). Pima Indians Diabetes Database. Kaggle. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

Shams, M. Y., Tarek, Z., & Elshewey, A. M. (2025, January 6). A novel RFE-GRU model for diabetes classification using Pima Indian Dataset. Nature News. https://www.nature.com/articles/s41598-024-82420-9#:~:text=Introducing%20a%20variety%20of%20machine,Histogram%20Gradient%20Boosting%20(HGB). 
 

