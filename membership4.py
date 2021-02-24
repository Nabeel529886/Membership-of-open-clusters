import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

raw_data = pd.read_csv("NGC2112(3040)(20).csv")
data = raw_data.fillna(raw_data.mean())
#print(data)

def normalize(dataset):
    norm_data = (dataset - dataset.mean())/dataset.std()
    return norm_data


source_id = data.loc[:, "source_id"]
ra = data.loc[:, "ra"]
dec = data.loc[:, "dec"]
pmra = data.loc[:, "pmra"]
pmdec = data.loc[:, "pmdec"]
parallax = data.loc[:, "parallax"]
phot_g_mean_mag = data.loc[:, "phot_g_mean_mag"]
bp_rp = data.loc[:, "bp_rp"]
#g_rp = data.loc[:, "g_rp"]


df_1 = pd.DataFrame(data=[ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp]).T
#print(df_1)
norm_df_1 = normalize(df_1)
#print(norm_dist_df_1)

gmm = GaussianMixture(n_components=2, covariance_type="full" ,n_init=1, init_params="kmeans", tol=1e-3, reg_covar=1e-6, verbose=2, verbose_interval=20)

fit_model_1 = gmm.fit(norm_df_1)

probs_1 = fit_model_1.predict_proba(norm_df_1)
prob_1_comp = [item[0] for item in probs_1]
#print(prob_1_comp)

prob_1_comp_series = pd.Series(prob_1_comp, name="probabilities")
df_1.loc[:, "probabilities"] = prob_1_comp_series
#print(prob_1_comp_series)
#source_id_list = source_id.tolist()
#prob_dict = dict(zip(source_id_list, prob_1_comp))
#print(prob_dict)
#print(len(prob_dict))
cluster = df_1.loc[:, "probabilities"] >= 0.8
cluster_df_1 = df_1[cluster]

field_df_1 = df_1[~cluster]
#print(cluster_df_1)
#print(cluster_df_1)
#df_2 = cluster_df_1.drop(columns=["probabilities"])
#print(df_2)
#norm_df_2 = normalize(df_2) 
#print(norm_df_2)
#gmm_2 = GaussianMixture(n_components=2, init_params="kmeans")
#print(df_2.shape)

#fit_model_2 = gmm_2.fit(norm_df_2)
#probs_2 = fit_model_2.predict_proba(norm_df_2)
#prob_2_comp = [item[0] for item in probs_2]
#print("length", len(prob_2_comp))
#print(prob_2_comp)

#prob_2_comp_series = pd.Series(prob_2_comp, name="probabilities")
#print(prob_2_comp_series)
#df_2 = df_2.assign(probabilities=prob_2_comp)
#print(df_2)
#df_2 = df_2.dropna()
#print(df_2)
#df_1_filtered = df_1[df_1.loc[:, "probabilities"] < 0.8]
#print(df_1_filtered)
#final_df = df_1_filtered.append(df_2)
#print(final_df)
#probabs = final_df.loc[:, "probabilities"]
#probabs_list = probabs.tolist()


#cluster = final_df.loc[:, "probabilities"] >= 0.8
#final_cluster_df = final_df[cluster]

#final_field_df = final_df[~cluster]

#clusters = []
#for prob in probabs_list:
 #   if prob >= 0.8:
  #      clusters.append(prob)
        
#source_id_f = final_df.loc[:, "source_id"]
'''
ra_f = final_df.loc[:, "ra"]
dec_f = final_df.loc[:, "dec"]
pmra_f = final_df.loc[:, "pmra"]
pmdec_f = final_df.loc[:, "pmdec"]
parallax_f = final_df.loc[:, "parallax"]
phot_g_mean_mag_f = final_df.loc[:, "phot_g_mean_mag"]
bp_rp_f = final_df.loc[:, "bp_rp"]
g_rp_f = final_df.loc[:, "g_rp"]
probabs_f = final_df.loc[:, "probabilities"]
'''        
#print(len(clusters))

'''
Plotting Starts From Here On
'''

##################START#######################
#Membership probability graph
'''
scatter = plt.scatter(phot_g_mean_mag_f, probabs_f, s=0.5, c=probabs_list, cmap="brg")
plt.xlabel("G Magnitude")
plt.ylabel("Probabilities")
plt.title("GMM Membership Probability")
'''
#####################END######################




######################START####################
#Photometric Analysis Graph

fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, sharex=True)
fig.suptitle("Photometric Analysis of NGC2112")
full_scatter = ax1.scatter(bp_rp, phot_g_mean_mag, s=0.2, c="black")
ax1.set_xlim(min(bp_rp), max(bp_rp))
ax1.set_ylim(max(phot_g_mean_mag), min(phot_g_mean_mag))
ax1.set(xlabel="G-RP mag", ylabel="G mag", title="All Stars")

cluster_scatter = ax2.scatter(cluster_df_1.loc[:, "bp_rp"], cluster_df_1.loc[:, "phot_g_mean_mag"], s=0.2, c="black")
ax2.set(xlabel="G-RP mag", ylabel="G mag", title="Cluster Stars")
field_scatter = ax3.scatter(field_df_1.loc[:, "bp_rp"], field_df_1.loc[:, "phot_g_mean_mag"], s=0.2, c="black")
ax3.set(xlabel="G-RP mag", ylabel="G mag", title="Field Stars")

#######################END###################




####################START######################
#Proper Motion Analysis Graph

fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, sharex=True)
fig.suptitle("Proper Motion Analysis")
cluster_scatter = ax1.scatter(cluster_df_1.loc[:,"pmra"], cluster_df_1.loc[:,"pmdec"], s=0.01, c="black")
ax1.set(xlabel="PMRA (mas/yr)", ylabel="PMDEC (mas/yr)", title="Cluster Stars")
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
field_scatter = ax2.scatter(field_df_1.loc[:, "pmra"], field_df_1.loc[:, "pmdec"], s=0.001, c="black")
ax2.set(xlabel="PMRA (mas/yr)", ylabel="PMDEC (mas/yr)", title="Field Stars")

##################END########################




##################START########################
# Ra and Dec graph

'''
scatter = plt.scatter(ra_f, dec_f, s=probabs_f + 0.2, c=probabs_f, cmap="brg")
plt.xlabel("RA (deg)")
plt.ylabel("DEC (deg)")
plt.title("RA and DEC Analysis")

'''
#################END#########################


plt.show()
