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

#Fit a 4 component model
pls = PLSRegression(n_components=4)
pls.fit(X_train_scaled, Y_train)

#Get X loadings
x_loadings = pd.DataFrame(pls_final.x_loadings_, index=X.columns, columns=[f'Component {i+1}' for i in range(4)])

#Get Y loadings
y_loadings = pd.DataFrame(pls_final.y_loadings_,index=['Diabetes_binary'],columns=[f'Component {i+1}' for i in range(4)])

