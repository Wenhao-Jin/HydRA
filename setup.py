from setuptools import setup
from setuptools import find_packages

#from distutils.extension import Extension

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "hydra_rbp",
    description ="A hybrid ensemble classifier for RNA-binding proteins utilizing protein-protein interaction context and protein sequence features.",
    long_description = long_description,
    long_description_content_type="text/markdown",
    version = "0.1.21.23",
    packages = find_packages(),
	package_data = {'HydRa': ['HydRa/data/*', 'HydRa/pre_trained/*']},
    install_requires = ['setuptools', 
                        'pandas',
                        'numpy',
                        'networkx >= 2.0',
                        'scikit-learn >= 0.22.1, < 0.23.1',
                        'tensorflow >= 2.3.1',
                        'keras >= 2.4.3, < 2.7.0',
                        'matplotlib',
                        'protein-bert'
                        ],
      
    setup_requires = ["setuptools_git >= 0.3",],
    entry_points={
                    'console_scripts':['HydRa_predict = HydRa.HydRa_predict:call_main', 'HydRa2_predict = HydRa.HydRa2_0_predict:call_main', 'HydRa_train = HydRa.training.train_HydRa:call_main', 'HydRa_train2 = HydRa.training.train_HydRa2_0:call_main', 'HydRa_train_eval = HydRa.training.train_and_evaluate_HydRa:call_main', 'HydRa2_train_eval = HydRa.training.train_and_evaluate_HydRa2_0:call_main', 'occlusion_map = HydRa.predictions.Occlusion_map_v2:call_main',  'occlusion_map2 = HydRa.predictions.Occlusion_map_v2_HydRa2_0:call_main','occlusion_map3 = HydRa.predictions.Occlusion_map_v3_HydRa2_0:call_main','test_data_load = HydRa.test_data_load:load_SS_list']
                    },
    #metadata for upload to PyPI
    author = "Wenhao Jin",
    author_email = "vincenzojin@gmail.com",
    keywords = "RNA-binding proteins, prediction, CNN, SVM, PPI, bioinformatics",
    url = "https://github.com/Wenhao-Jin/HydRa",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    
    #Other stuff I feel like including here
    include_package_data = True
    #zip_safe = True #True I think
)
