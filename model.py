import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# PCA
# opening and storing the dataset
heart_failure = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# target of the dataset is DEATH_EVENT (value = either 0 or 1). In features, we drop the target variable
target = heart_failure['DEATH_EVENT']
features = heart_failure.drop('DEATH_EVENT', axis=1)

# scaling the features
features_scaled = (features - features.mean()) / features.std()

# after testing, we chose 9 as the value of n_components
# applying PCA on the dataset
pca = PCA(n_components=9)
pca.fit(features_scaled)
transformed_features = pca.transform(features_scaled)
heart_failure_pca = pd.DataFrame(transformed_features, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])

# ----------------------------------------------------------------------------------------------------------------------------------

# SVM
model = SVC()
model.fit(heart_failure_pca, target)

# user_input only used for testing; replace the list with real answers
user_input = [62.0, 1, 447, 1, 30, 1, 265000.00, 2.5, 132, 1, 1, 7]
user_input_df = pd.DataFrame(columns=['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'])
user_input_df.loc[len(user_input_df)] = user_input

# reducing dimensionality of user_input using the same PCA
user_input_transformed = pca.transform(user_input_df)
user_input_pca = pd.DataFrame(user_input_transformed, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])

# prediction
prediction = model.predict(user_input_pca)

print(prediction)