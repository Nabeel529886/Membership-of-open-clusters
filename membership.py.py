import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt


def normalize(dataset):
        normalized_dataset = (dataset - dataset.mean())/dataset.std()
        return normalized_dataset

raw_data = pd.read_csv("NGC2112.csv")
data = raw_data.dropna()
data.to_csv("NGC2112_cleaned.csv", index=False)

norm_data = normalize(data)

gmm = GMM(n_components=2, covariance_type="full")

gmm.fit(norm_data)

data_labels = gmm.predict(norm_data)
df_data_labels = pd.Series(data_labels)

print(df_data_labels.count())



#probability = gmm.predict_proba(data)
#df_probability = pd.DataFrame(probability, columns=["probability 1", "probability 2"])
#print(df_probability.head())


main_df = pd.DataFrame(data)
main_df["clusters"] = df_data_labels
main_df_cleaned = main_df.dropna()
main_df_cleaned.to_csv("output2112.csv", index=False)


color = ["blue", "red"]

for i in range(0,2):
    data = main_df[main_df["clusters"] == i]
    plt.scatter(data["ra"], data["dec"], s=3,  c=color[i], alpha=0.6)
plt.title("Membership of NGC2112")
plt.xlabel("RA (deg)")
plt.ylabel("DEC (deg)")
plt.show()
