from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(67)
import numpy as np
np.random.seed(67)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

url = "https://raw.githubusercontent.com/IsaacBFine/diabetes-predictors/refs/heads/main/Data/clean/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

data = pd.read_csv(url)
data.head()
#Loads in dataset

#Split data into indicators and diabetes outcome.

X = data.drop(columns = ["Diabetes_binary"])
Y = data["Diabetes_binary"]

# Split data into training (80%) and testing (20%) sets
# random_state=67 ensures reproducibility and that we can debug with tests
# stratify=Y preserves the same 50/50 diabetes ratio in both train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=67, stratify=Y)
     

#Standardize data so binary and continuous variables have equal weight
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  #Calculate mean and standard deviation from training data then standardize it
X_test_scaled = scaler.transform(X_test)        #Apply same scale to test data

#Fit a 4 component model
pls = PLSRegression(n_components=4)
pls.fit(X_train_scaled, Y_train)

#Get X loadings
x_loadings = pd.DataFrame(pls_final.x_loadings_, index=X_train.columns, columns=[f'Component {i+1}' for i in range(4)])

#Get Y loadings
y_loadings = pd.DataFrame(pls_final.y_loadings_,index=['Diabetes_binary'],columns=[f'Component {i+1}' for i in range(4)])
plt.figure(figsize=(8, 6))

#Plot indicators
plt.scatter(x_loadings['Component 1'], x_loadings['Component 2'], color='blue')
for var in x_loadings.index:
    plt.annotate(var, (x_loadings['Component 1'][var], x_loadings['Component 2'][var]), fontsize=8)

#Plot diabetes
plt.scatter(y_loadings['Component 1'], y_loadings['Component 2'], color='orange', s=400, marker='*', label='Diabetes')
plt.annotate('Diabetes', (y_loadings['Component 1'].iloc[0], y_loadings['Component 2'].iloc[0]), fontsize=15, color='black', fontweight='bold')

plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel('Component 1 Loadings')
plt.ylabel('Component 2 Loadings')
plt.title('PLSR Loadings: Component 1 vs Component 2')
plt.show()
