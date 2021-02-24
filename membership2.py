import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GM
from sklearn import preprocessing
import seaborn as sns


raw_data = pd.read_csv("NGC2112(3040).csv")
data = raw_data.fillna(raw_data.mean())


col_names = data.columns
#print(col_names)


scaler = preprocessing.StandardScaler()

scaled_df = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaled_df, columns=col_names)
print(data)

gmm = GM(n_components=2, covariance_type="full", random_state=983, init_params="kmeans")

fitted_model = gmm.fit(scaled_df)

prob = fitted_model.predict_proba(scaled_df)
prob_first_comp = [item[0] for item in prob]
print(len(prob_first_comp))

clusters = []
for prob in prob_first_comp:
    if prob >= 0.8:
        clusters.append(prob)
     
print(len(clusters))

radec = scaled_df.loc[:,"dec"]
print(radec)

scatter_plot = plt.scatter(data.loc[:, "pmra"], data.loc[:, "pmdec"], c=prob_first_comp, s=0.08, cmap="brg")
#plt.hist(prob_first_comp)
#sns.distplot(radec)
#plt.colorbar(scatter_plot)
plt.xlim(-7,7)
plt.ylim(-7, 7)
plt.tight_layout()
#plt.ylim()
plt.show()

