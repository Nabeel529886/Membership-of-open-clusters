import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from sklearn import preprocessing


raw_data = pd.read_csv("NGC2112(3040).csv")

data = raw_data.fillna(raw_data.mean())

source_id = data.loc[:, "source_id"]
ra = data.loc[:, "ra"]
dec = data.loc[:, "dec"]
pmra = data.loc[:, "pmra"]
pmdec = data.loc[:, "pmdec"]
parallax = data.loc[:, "parallax"]
phot_g_mean_mag = data.loc[:, "phot_g_mean_mag"]
g_rp = data.loc[:, "g_rp"]
bp_rp = data.loc[:, "bp_rp"]



def normalize(dataset):
    norm_data = (dataset - dataset.mean())/ dataset.std()
    return norm_data

dist_df = pd.DataFrame(data = [ra, dec, pmra, pmdec, parallax]).T
#print(dist_df)

col_names = dist_df.columns

#scaler = preprocessing.StandardScaler()
scaled_df = normalize(dist_df)
#scaled_df = scaler.fit_transform(dist_df)
#print(scaled_df)



gmm = GaussianMixture(n_components=2, max_iter=3040, covariance_type="full", init_params="kmeans", random_state=127)
fit_model = gmm.fit(scaled_df)
#print(gmm.means_)
#print(scaled_df)

probs = fit_model.predict_proba(scaled_df)



#print(probs)
prob_comp = [item[1] for item in probs]
#print(len(prob_comp))

prob_comp_series = pd.Series(prob_comp, name="Probabilites")
#print(prob_comp_series)
dist_df.loc[:, "probabilities"] = prob_comp_series

cluster = dist_df.loc[:, "probabilities"] >= 0.8
cluster_df = dist_df[cluster]
#print(cluster_df)

cluster_df.loc[:, "g_rp"] = g_rp[cluster]
cluster_df.loc[:, "phot_g_mean_mag"] = phot_g_mean_mag[cluster]
#print(cluster_df)

field_df = dist_df[~cluster]
#print(cluster_df)

field_df.loc[:, "g_rp"] = g_rp[~cluster]
field_df.loc[:, "phot_g_mean_mag"] = phot_g_mean_mag[~cluster]


#print(len(ra))

clusters = []
for prob in probs:
    if prob[1] >= 0.9:
        clusters.append(prob)


         
#print(len(clusters))
#print(gmm.means_)

#Membership probability graph
'''
scatter = plt.scatter(phot_g_mean_mag, prob_comp_series, s=0.5, c=prob_comp, cmap="brg")
plt.xlabel("G Magnitude")
plt.ylabel("Probabilities")
plt.title("GMM Membership Probability")
'''

#scatter = plt.scatter(cluster_df.loc[:, "g_rp"], cluster_df.loc[:, "phot_g_mean_mag"], s=0.05 ,alpha=1, c='black', cmap="brg")
#plt.title("Photometric Analysis")
#plt.xlabel("G-RP Mag")
#plt.ylabel("G Mag")

#Photometric Analysis Graph
'''
fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, sharex=True)
fig.suptitle("Photometric Analysis of NGC2112")
full_scatter = ax1.scatter(g_rp, phot_g_mean_mag, s=0.2, c="black")
ax1.set_xlim(min(g_rp), max(g_rp))
ax1.set_ylim(max(phot_g_mean_mag), min(phot_g_mean_mag))
ax1.set(xlabel="G-RP mag", ylabel="G mag", title="All Stars")

cluster_scatter = ax2.scatter(cluster_df.loc[:, "g_rp"], cluster_df.loc[:, "phot_g_mean_mag"], s=0.2, c="black")
ax2.set(xlabel="G-RP mag", ylabel="G mag", title="Cluster Stars")
field_scatter = ax3.scatter(field_df.loc[:, "g_rp"], field_df.loc[:, "phot_g_mean_mag"], s=0.2, c="black")
ax3.set(xlabel="G-RP mag", ylabel="G mag", title="Field Stars")
'''

#Proper Motion Analysis Graph

scatter = plt.scatter(pmra, pmdec, s=0.5, c=prob_comp, cmap="brg")
plt.title("Proper Motion Analysis")
plt.xlabel("PMRA (mas/yr)")
plt.ylabel("PMDEC (mas/yr)")
plt.xlim(-7, 7)
plt.ylim(-7, 7)


# Ra and Dec graph
'''
scatter = plt.scatter(ra, dec, s=prob_comp_series + 0.5, c=prob_comp, cmap="brg")
plt.xlabel("RA (deg)")
plt.ylabel("DEC (deg)")
plt.title("RA and DEC Analysis")
plt.colorbar(scatter)
'''

#
plt.tight_layout()
plt.show()
