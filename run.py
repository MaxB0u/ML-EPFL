from src.helpers import *


path_dataset_tr = './dataset/train.csv'

x_col_names = ["DER_mass_MMC",
               "DER_mass_transverse_met_lep",
               "DER_mass_vis",
               "DER_pt_h",
               "DER_deltaeta_jet_jet",
               "DER_mass_jet_jet",
               "DER_prodeta_jet_jet",
               "DER_deltar_tau_lep",
               "DER_pt_tot",
               "DER_sum_pt",
               "DER_pt_ratio_lep_tau",
               "DER_met_phi_centrality",
               "DER_lep_eta_centrality",
               "PRI_tau_pt",
               "PRI_tau_eta",
               "PRI_tau_phi",
               "PRI_lep_pt",
               "PRI_lep_eta",
               "PRI_lep_phi",
               "PRI_met",
               "PRI_met_phi",
               "PRI_met_sumet",
               "PRI_jet_num",
               "PRI_jet_leading_pt",
               "PRI_jet_leading_eta",
               "PRI_jet_leading_phi",
               "PRI_jet_subleading_pt",
               "PRI_jet_subleading_eta",
               "PRI_jet_subleading_phi",
               "PRI_jet_all_pt"]

y_col_name = ['Prediction']

x_raw, y = load_data(path_dataset_tr, x_col_names, y_col_name)
x = standardize(x_raw)
tx = build_model_data(x, y)

#call the models