import pkg_resources
import pandas as pd

def load_SS_list():
	stream = pkg_resources.resource_stream(__name__, 'data/SVM_SS_11mer_chi2_alpha0.01_with_LinearSVC_selected_features_From_WholeDataSet.txt')
	return pd.read_csv(stream)


