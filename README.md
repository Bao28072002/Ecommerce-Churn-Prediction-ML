# Churn-Prediction-Segmentation-For-Retention-Strategy-For-Ecommerce-Machine-Learning---Python

<img width="1024" height="640" alt="image" src="https://github.com/user-attachments/assets/c502cb1a-4cef-4f1e-a16b-d2d22b2656be" />

**Author:** Lê Gia Bảo

**Date:** October 2025

**Tools Used:** Machine Learning - Python 


## 📘 Table of Contents
1. [📌 Background & Overview](#-background--overview)  
2. [📂 Dataset Description & Data Structure](#-dataset-description--data-structure)  
3. 🌱 [Data Preprocessing](#-data-preprocessing)  
4. 🔍 [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)  
5. 📊 [Train & Apply Churn Prediction Model](#-train--apply-churn-prediction-model)  
6. 💡 [Key Findings and Recommendations for Retention](#-key-findings-and-recommendations-for-retention)  
7. 🤖 [Create A Model For Predicting Churn](#-create-a-model-for-predicting-churn)  
8. 🧑‍💻 [Customer Segmentation Using Clustering](#-customer-segmentation-using-clustering)  

---

## 📌 Background & Overview

### 🎯 **Objective**

This project aims to **predict and segment churned customers** in an e-commerce business to **build effective retention strategies**.  
By leveraging **Machine Learning and Python**, it focuses on:

- 🔍 Identifying key behaviors and patterns of churned users  
- 🤖 Developing a predictive model to forecast customer churn  
- 🎯 Segmenting churned users to tailor personalized offers and promotions  

---

### ❓ **Business Questions Addressed**

- What are the main factors driving customer churn in e-commerce?  
- How can we accurately predict potential churners and take preventive action?  
- What steps are needed to build a reliable churn prediction model?  
- How can churned customers be segmented for targeted marketing campaigns?  

---

### 👤 **Intended Audience**

- **Data & Business Analysts** – To uncover insights into churn behavior and retention drivers  
- **Marketing & Retention Teams** – To design data-driven promotional strategies  
- **Executives & Decision-Makers** – To reduce churn and enhance customer lifetime value
  
---

## 📂 Dataset Description & Data Structure

### 📌 **Data Source**  
**Source:** The dataset is obtained from the e-commerce company's database.  
**Size:** The dataset contains 5,630 rows and 20 columns.  
**Format:** .xlxs file format.

### 📊 **Data Structure & Relationships**

1️⃣ **Tables Used:**  
The dataset contains only **1 table** with customer and transaction-related data.

2️⃣ **Table Schema & Data Snapshot**  
**Table: Customer Churn Data**

<details>
  <summary>Click to expand the table schema</summary>

| **Column Name**              | **Data Type** | **Description**                                              |
|------------------------------|---------------|--------------------------------------------------------------|
| CustomerID                   | INT           | Unique identifier for each customer                          |
| Churn                        | INT           | Churn flag (1 if customer churned, 0 if active)              |
| Tenure                       | FLOAT         | Duration of customer's relationship with the company (months)|
| PreferredLoginDevice         | OBJECT        | Device used for login (e.g., Mobile, Desktop)                 |
| CityTier                     | INT           | City tier (1: Tier 1, 2: Tier 2, 3: Tier 3)                   |
| WarehouseToHome              | FLOAT         | Distance between warehouse and customer's home (km)         |
| PreferredPaymentMode         | OBJECT        | Payment method preferred by customer (e.g., Credit Card)     |
| Gender                       | OBJECT        | Gender of the customer (e.g., Male, Female)                  |
| HourSpendOnApp               | FLOAT         | Hours spent on app or website in the past month              |
| NumberOfDeviceRegistered     | INT           | Number of devices registered under the customer's account   |
| PreferedOrderCat             | OBJECT        | Preferred order category for the customer (e.g., Electronics)|
| SatisfactionScore            | INT           | Satisfaction rating given by the customer                    |
| MaritalStatus                | OBJECT        | Marital status of the customer (e.g., Single, Married)       |
| NumberOfAddress              | INT           | Number of addresses registered by the customer               |
| Complain                     | INT           | Indicator if the customer made a complaint (1 = Yes)         |
| OrderAmountHikeFromLastYear  | FLOAT         | Percentage increase in order amount compared to last year   |
| CouponUsed                   | FLOAT         | Number of coupons used by the customer last month            |
| OrderCount                   | FLOAT         | Number of orders placed by the customer last month           |
| DaySinceLastOrder            | FLOAT         | Days since the last order was placed by the customer        |
| CashbackAmount               | FLOAT         | Average cashback received by the customer in the past month  |

</details>

---

## 🌱 Data Preprocessing

📌 Import Necessary Libraries
[In 1]: 

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
```
[In 2]: 

```python
# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# Define the path to the project folder
data='/content/drive/MyDrive/Portfolio/Machine Learning/final_project/churn_prediction.xlsx'

# Load the data
df = pd.read_excel(data)
```
[In 3]:

```python
df.head(5)
```
[Out 3]:

<img width="1774" height="215" alt="image" src="https://github.com/user-attachments/assets/73ce54f8-7b42-442d-b532-72e39635a259" />

[In 4]:

```python
df.info()
```
[Out 4]:

<img width="480" height="404" alt="image" src="https://github.com/user-attachments/assets/840e3bb5-b648-44ff-881e-a2644af47467" />

[In 5]:

```python
#check missing value
missing_value = df.isna().sum()
print(missing_value)
```
[Out 5]:

<img width="323" height="318" alt="image" src="https://github.com/user-attachments/assets/d935cf53-53b4-40da-9cc1-c386837575c4" />

[In 6]:

```python
# Check for duplicate
check_dup = df.duplicated().sum()
check_dup
```
[Out 6]:

<img width="93" height="35" alt="image" src="https://github.com/user-attachments/assets/1bd35029-5bea-4d66-9982-065d4a04d9da" />

#### 💡 **Data Understanding**

📌 Before conducting any analysis or model training, several key steps were performed to **explore and preprocess the dataset**.

---

### 📝 **1. Checked Dataset Structure**

An initial inspection of the dataset provided an overview of the **number of rows, columns, and data types** for each feature, along with summary statistics.

- The dataset contains **5,630 rows** and **20 columns**, consisting of both **numerical and categorical variables**.  
- **Missing values** were identified in multiple columns, including `Tenure`, `WarehouseToHome`, `HourSpendOnApp`, and others.

---

### 🧩 **2. Checked for Missing Values**

Several features had missing data:

| Column | Missing Values |
|:--------|:----------------:|
| `Tenure` | 264 |
| `WarehouseToHome` | 251 |
| `HourSpendOnApp` | 255 |
| `OrderAmountHikeFromlastYear` | 265 |
| `CouponUsed` | 256 |
| `OrderCount` | 258 |
| `DaySinceLastOrder` | 307 |

---

### 🔍 **3. Checked for Duplicates**

After checking for duplicate records, it was confirmed that **no duplicate entries** exist in the dataset.

---

### 📊 **Summary**

- Several columns contained **missing values**, which were handled by **replacing them with their respective mean values** to prepare the data for analysis and modeling.  
- Some categorical features contained **inconsistent text representations** (different spellings of the same value). These were **standardized** to ensure data consistency.  

**📝 Missing Value Handling**

[In 7]:

```python
# Define the list of columns with missing values
cols_missing = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']

# Replace missing columns with median
for col in cols_missing:
    # Fill missing values in each column with the median of that column
    df[col].fillna(value= df[col].median(), inplace=True)
```
---

## 🔍 Exploratory Data Analysis (EDA)

Analyzed the distribution of continuous features to understand their **uniqueness and variability**.  
Most continuous variables showed a limited range of unique values, which is **reasonable within the dataset’s business context**.

---

### 🔍 **Univariate Analysis**

- **Categorical Variables:**  
  Variables such as `PreferredLoginDevice`, `Gender`, `MaritalStatus`, and others were explored using **count plots**.  
  This helped visualize the **distribution and balance of each category** across the dataset.  
  Additionally, we examined the **relationship between categorical variables and churn**. Count plots with `Churn` as hue highlighted which categories had higher churn rates, providing insights into potential risk groups.

- **Continuous Variables:**  
  Features like `Tenure`, `SatisfactionScore`, and `CashbackAmount` were analyzed using **boxplots** to detect potential outliers.  
  Although outliers were identified, they were considered **meaningful** and **retained**, as they likely represent **distinct customer behaviors important for churn prediction**.  
  We also visualized these continuous variables against `Churn` to observe differences in distributions between churned and non-churned customers. For example, churned customers tended to have **lower `SatisfactionScore`** and **shorter `Tenure`**, indicating these features are strong predictors for churn.

  [In 8]:

```python
df.drop('CustomerID', axis=1, inplace=True)
sns.set(style="darkgrid", palette="muted")
fig, ax = plt.subplots(figsize=(16,7))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.show()
```
[Out 8]:

<img width="1221" height="689" alt="image" src="https://github.com/user-attachments/assets/903a24b8-592a-4420-b03a-477fa9f22310" />

[In 9]:

```python
fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(26,12))
for col,subplot in zip(cat, ax.flatten()):
    sns.countplot(x = df[col], hue=df.Churn, ax=subplot)
```
[Out 9]:

<img width="1630" height="750" alt="image" src="https://github.com/user-attachments/assets/d53496bc-aa3b-40a7-adaa-45c763beb78e" />

[In 10]:

```python
fig, ax = plt.subplots(2, 3, figsize=(30, 20))
plt.rcParams['font.size'] = 16
for col, subplot in zip(cat, ax.flatten()):
    churn_count = df.groupby(col)['Churn'].sum()
    total_count = df[col].value_counts().sort_index()
    churn_rate = (churn_count / total_count) * 100
    subplot.pie(
        labels=churn_rate.index,
        x=churn_rate.values,
        autopct='%.0f%%',
        textprops={'fontsize': 16}
    )
    subplot.set_title(f"Churn % theo {col}")
```
[Out 10]:

<img width="1127" height="738" alt="image" src="https://github.com/user-attachments/assets/d9af9a93-b644-4127-83fd-446dddcf48b1" />

- **Insights:**  
  - Certain categories in categorical variables show **higher churn rates**, highlighting target groups for retention strategies.  
  - Continuous variables display **distinct patterns** between churned and non-churned customers, supporting their importance in predictive modeling.


## 📊 Train & Apply Churn Prediction Model

### **📝 Encoding**

### 🔧 Preprocessing Steps

**Encoding Categorical Features:**  
- **One-Hot Encoding:**  
  Categorical columns with a small number of unique values were encoded using **one-hot encoding**. This generates binary columns for each unique category, allowing the model to better process categorical data. The columns encoded with one-hot encoding are:  
  - `PreferredLoginDevice`  
  - `PreferredPaymentMode`  
  - `PreferedOrderCat`  
  - `MaritalStatus`  

- **Label Encoding:**  
  The `Gender` column was encoded using **label encoding**, converting categorical labels into numerical values (`0` or `1`).  

**Dropping Unnecessary Columns:**  
- The `CustomerID` column was removed in `[In 8]` as it is a unique identifier and does not provide predictive value for the model.

[In 11]:

```python
df_encoded = pd.get_dummies(df, columns=['PreferredLoginDevice', 'PreferredPaymentMode', 'PreferedOrderCat', 'MaritalStatus', 'PreferredLoginDevice'])
label_encoder = LabelEncoder()
df_encoded['Gender'] = label_encoder.fit_transform(df_encoded['Gender'])
```

[Out 11]:

<img width="1770" height="300" alt="image" src="https://github.com/user-attachments/assets/019f2613-50e9-4391-8552-98a077065dd4" />

### **📝 Split Data into Features (X) and Target (y)**

[In 8]:

- The dataset was split into **features (X)** and **target (y)**, where:
  - **X** contains all the independent variables (features), and
  - **y** contains the target variable `Churn`.

```python
x = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']  # Target
# (80/20 split)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
[In 9]:

### 📝 Standardizing Features with StandardScaler

- All features were standardized using **StandardScaler**, ensuring a **mean of 0** and a **standard deviation of 1**. Standardization is crucial because many machine learning algorithms perform better when features are on the same scale.  
- The **training set** was both **fitted and transformed**, while the **test set** was **only transformed** to prevent data leakage.
  
```python
# Standardize the features using StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
```
### 📝 Applying the Model – Random Forest Classifier

- A **Random Forest Classifier** was trained on the standardized features.  
- The model’s performance was evaluated using **accuracy** on both the **training** and **test** sets.
  
[In 10]:

```python
# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# Make predictions on training and test sets
y_pred_train = model.predict(x_train_scaled)
y_pred_test = model.predict(x_test_scaled)

# Evaluate model accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
train_balanced_acc = balanced_accuracy_score(y_train, y_pred_train)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

# Print the results
print(f'Training Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Training Balanced Accuracy: {train_balanced_acc:.4f}')
print(f'Test Balanced Accuracy: {test_balanced_acc:.4f}')
```
### 💡 Conclusion

The evaluation of the model's performance on both the training and test sets is as follows:

- **Training Accuracy:** 1.0000  
  - The model achieves perfect accuracy on the training set. This indicates that the model has learned the training data very well, but it also suggests a potential risk of overfitting.
  
- **Test Accuracy:** 0.9574  
  - The model performs very well on the test set, with a high accuracy of 95.74%. This indicates that the model generalizes well to unseen data, with only a slight drop from the training accuracy.

- **Training Balanced Accuracy:** 1.0000  
  - The balanced accuracy on the training set is perfect, showing that the model predicts both classes (Churn and Non-Churn) equally well on the training data.

- **Test Balanced Accuracy:** 0.9002  
  - The balanced accuracy on the test set is 90.02%, indicating strong performance even for potentially imbalanced classes.

➡ Overall, the model performs excellently, with **high accuracy** and **balanced accuracy** on both training and test sets. While the perfect training accuracy suggests a **risk of overfitting**, the high test accuracy and balanced accuracy demonstrate that the model still **generalizes effectively**. Continuous monitoring on additional datasets is recommended to ensure **robustness in real-world scenarios**.

### 📝 **Apply Random Forest To Find Important Features**

## 1️⃣ Define Parameter Grid for Hyperparameter Tuning
A grid of hyperparameters for the **Random Forest Classifier** was defined to explore different combinations:

- `n_estimators`: number of trees in the forest (`50`, `100`)  
- `max_depth`: maximum depth of each tree (`None`, `10`, `20`)  
- `min_samples_split`: minimum samples required to split a node (`2`, `5`)  
- `min_samples_leaf`: minimum samples required at a leaf node (`1`, `2`)  
- `bootstrap`: whether bootstrap samples are used (`True`)  

This grid allows us to systematically test combinations of hyperparameters to find the most optimal model.

---

## 2️⃣ Perform Grid Search with Cross-Validation
- **`GridSearchCV`** was used to find the best combination of hyperparameters with **5-fold cross-validation**.  
- The scoring metric was set to **balanced accuracy** to account for possible class imbalance.  
- After fitting, the best parameters found were:

## 3️⃣ Evaluate the Best Model

- The best estimator from `GridSearchCV` was used to make predictions on both the **training** and **test** sets.  
- **Balanced accuracy** was calculated to account for potential class imbalance:

| Dataset  | Balanced Accuracy |
|----------|-----------------|
| Training | 1.0             |
| Test     | 0.892           |

- **Interpretation:**  
  - The training accuracy of 1.0 shows that the model has perfectly learned the training data, which may indicate **overfitting**.  
  - The test accuracy of 0.892 indicates the model **generalizes well** to unseen data, with only a slight drop from the training set.

---

## 4️⃣ Feature Importance Analysis

- The feature importances of the best Random Forest model were extracted using `feature_importances_`.  
- Each feature and its corresponding **Gini importance** were stored in a DataFrame and sorted in ascending order.  
- A horizontal bar chart was plotted to display the **top 20 most important features**:

[In 11]:

```python
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(x_test.columns, best_clf.feature_importances_):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True)

importances = importances.reset_index()

# Create bar chart
plt.figure(figsize=(10, 10))
plt.barh(importances.tail(20)['index'][:20], importances.tail(20)['Gini-importance'])

plt.title('Feature Important')

# Show plot
plt.show()
```
[Out 11]:

<img width="789" height="615" alt="image" src="https://github.com/user-attachments/assets/29a51039-8573-4721-9196-f88aecf816c5" />

### **💡Conclusion: **

Based on the image, we can conclude that the **top 5 most important features** directly affecting churn behavior are: **Tenure**, **CashbackAmount**, **WarehouseToHome**, **Complain**, and **DaySinceLastOrder**.

---

## 🎯 The Role of the Features

These features play a **crucial role** in predicting whether a customer is likely to churn.

* **Tenure** and **Days Since Last Order** indicate customer **engagement**.
* **Cashback Amount** and **Complaints** reflect **satisfaction levels**.
* **Warehouse to Home Distance** may influence the **delivery experience**, impacting customer retention.

---

## 📈 Visualizing Churn Behavior

Next, we will plot a **histogram chart** to visualize the differences between churn and non-churn behavior for these top important features.

This will help us **identify patterns** and **understand** how these features contribute to customer churn.

---

## 💡 Key Findings and Recommendations for Retention

## 📉 Churn Feature Insights

| Metric | Churn (Blue) | Non-Churn (Red) | Insight |
| :--- | :--- | :--- | :--- |
| **Tenure** (Customer Lifetime) | 80% left within 5 months, very few stayed beyond 10 months | More evenly distributed, many customers stayed over 20 months | **Churning customers leave very early**, meaning the initial experience is crucial. Without effective retention strategies, they leave quickly. |
| **CashbackAmount** (Cashback Received) | Average around 100–200, wide distribution | Mainly concentrated at 120–250 | Churn customers received **less cashback**, resulting in a lower perceived financial benefit. Increasing cashback or offering alternative incentives is necessary to improve retention. |
| **WarehouseToHome** (Delivery Time/Distance) | Wide distribution, average 15–30 days, many orders exceeded 35 days | Mostly under 20 days, rarely over 25 days | **Longer delivery times are correlated with higher churn.** Optimizing logistics and reducing shipping time can improve retention. |
| **Complain** (Customer Complaint) | Either 0 or 1, ratio ~50%-50% | Similar but the majority of customers did not complain, complaints only about 10–15% | Churn customers have a **higher complaint rate**, but many also leave without complaining. This suggests the need to proactively reach out to dissatisfied customers instead of waiting for feedback. |
| **DaySinceLastOrder** (Days Since Last Order) | Wide distribution, many >10 days, many cases exceeding 20 days | Usually under 10 days, rarely over 15 days | Churn customers **order less frequently**. If they haven't purchased for a long time, the risk of churn is high. Re-engagement campaigns (e.g., discounts for the next purchase) should be implemented. |

---

## 🤖 Create A Model For Predicting Churn

The model will be trained using the **top 5 features** impacting churn behavior:

* **Tenure** 
* **CashbackAmount** 
* **WarehouseToHome** 
* **Complain** 
* **DaySinceLastOrder** 

[In 12]:

```python
# Select top features affecting Churn
top_features = ['Tenure', 'CashbackAmount', 'WarehouseToHome', 'Complain', 'DaySinceLastOrder']
x_1 = df[top_features]
y_1 = df['Churn']

# Split: 70% train, 30% temp (val + test)
x_train1, x_val1, y_train1, y_val1 = train_test_split(x_1, y_1, test_size=0.3, random_state=42)

# Split temp into 15% val, 15% test
x_val1, x_test1, y_val1, y_test1 = train_test_split(x_val1, y_val1, test_size=0.5, random_state=42)

# Check dataset sizes
print(f"Train: {x_train1.shape}, Val: {x_val1.shape}, Test: {x_test1.shape}")

#Normalize data
from sklearn.preprocessing import StandardScaler
scaler_1 = StandardScaler()
x_train1_scaled = scaler.fit_transform(x_train1)
x_val1_scaled = scaler.transform(x_val1)
x_test1_scaled = scaler.transform(x_test1)
```
### 📝 **Choose Model**

**1. Choose metric for evaluating**

## 🎯 Why Recall is the Chosen Primary Metric

The decision to use **Recall** (also known as Sensitivity or True Positive Rate) as the primary evaluation metric is based entirely on the specific business objective in churn prediction: **retaining customers**.

---

### 1. Primary Goal: Accurately Identify Churners

The core business objective is to **accurately identify customers likely to churn** so that retention strategies (like special offers, personalized outreach, or service improvements) can be implemented. This requires the model to be **sensitive** to the "churn" class.

---

### 2. The High Cost of False Negatives (FN)

In churn prediction, the cost of making a **False Negative (FN)** error is significantly higher than the cost of a **False Positive (FP)** error.

* **False Negative (FN):** Occurs when the model predicts a customer will **NOT churn**, but they **ACTUALLY CHURN**.
    * **Impact:** The business **loses a customer** who could have been saved, resulting in the loss of all future revenue from that customer. This is the **most damaging type of error**.
* **False Positive (FP):** Occurs when the model predicts a customer **WILL churn**, but they **ACTUALLY DON'T**.
    * **Impact:** The business mistakenly targets a loyal customer with a retention offer. While this incurs a cost (the offer's expense, marketing resources), the customer is not lost.

Since **losing a customer (FN) causes more damage** than the expense of a mistaken retention effort (FP), the priority is to **minimize FN**.

**2. Model Comparison & Selection**  

| Model                  | Recall Score |
|------------------------|--------------|
| **Random Forest**      | **0.6979**    |
| Logistic Regression   | 0.3288       |
| KNN                   | 0.4697       |
| Gradient Boosting     | 0.6241     |

<img width="900" height="545" alt="image" src="https://github.com/user-attachments/assets/50e87593-5a01-4285-b2cd-af786675fc77" />

Based on the test results, the **Random Forest** model achieved the **highest Recall score**.

→ Therefore, we will **select Random Forest and proceed with fine-tuning** to further enhance the model's performance.

**3. Apply Model & Fine tune**

[In 13]: 

Apply **Random Forest** 

```python
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'bootstrap': [True],
    'class_weight': ['balanced', None]
}
rf_finetune = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='recall')
rf_finetune.fit(x_train1_scaled, y_train1)
print("Best parameters found:", rf_finetune.best_params_)
print("Best recall score:", rf_finetune.best_score_)
```

[Out 13]: 

<img width="927" height="31" alt="image" src="https://github.com/user-attachments/assets/785e9cae-a368-41a6-bfeb-991bbc70b20d" />

[In 14]: 

**Calculate the confusion matrix on the test set** 

```python
cm = confusion_matrix(y_test1, y1_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
[Out 14]: 

<img width="525" height="424" alt="image" src="https://github.com/user-attachments/assets/ad428981-5402-4bf5-9074-cf254c386883" />

| Predicted Class | Not Churn | Churn | Interpretation |
| :--- | :---: | :---: | :--- |
| **Actual: Not Churn** | **614 (TN)** | **78 (FP)** | The model correctly predicted 614 customers would stay. |
| **Actual: Churn** | **36 (FN)** | **117 (TP)** | The model correctly predicted 117 customers would leave. |

* **TN (True Negatives):** 614 correctly predicted "Not Churn".
* **TP (True Positives):** 117 correctly predicted "Churn".
* **FP (False Positives):** 78 incorrectly predicted "Churn" (they actually stayed).
* **FN (False Negatives):** 36 incorrectly predicted "Not Churn" (they actually left).

### 2. Key Performance Metrics

The following metrics are calculated considering "Churn" as the positive class of interest:

* **Accuracy:**
    * **86.4%** $\left(\frac{614 + 117}{845}\right)$
    * *Explanation:* The overall ratio of correct predictions across all data.
* **Recall / Sensitivity:**
    * **76.5%** $\left(\frac{117}{117 + 36}\right)$
    * *Explanation:* Out of the **153** customers who **actually churned**, the model successfully identified **76.5%** of them.
* **Precision:**
    * **60.0%** $\left(\frac{117}{117 + 78}\right)$
    * *Explanation:* When the model predicts a customer will churn, that prediction is correct only **60%** of the time.

### 3. Overall Assessment

The model demonstrates **good overall Accuracy (86.4%)**. However, **Precision (60.0%)** is significantly lower than **Recall (76.5%)**, indicating that while the model finds most of the actual churners, a high proportion of its "Churn" predictions are incorrect (False Positives).
---

## 🧑‍💻 Customer Segmentation Using Clustering
Phân khúc khách hàng dựa trên hành vi và đặc điểm chung.



