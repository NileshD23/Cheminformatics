# This was based on: https://pubs.acs.org/doi/10.1021/ci034243x


# Install PyCaret in terminal
# pip install pycaret

# === Read in Dataset === 
import pandas as pd

delaney_with_descriptors_url = 'https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv'
dataset = pd.read_csv(delaney_with_descriptors_url)

print(dataset)


# === Model Building
# Model Setup
from pycaret.regression import *

model = setup(data = dataset, target = 'logS', train_size=0.8, silent=True)

# Model Comparison
compare_models()

# Model Creation
et = create_model('et')

# Model Tuning
tuned_et = tune_model('et', n_iter = 50, optimize = 'mae')
print(tuned_et)

# Residual Plot
plot_model(et, 'residuals')

# Prediction Error Plot
plot_model(et, 'error')

# Cooks Distance Plot
plot_model(et, 'cooks')

# Recursive Feature Selection
plot_model(et, 'rfe')

# Learning Curve
plot_model(et, 'learning')

# Validation Curve
plot_model(et, 'vc')

# Manifold Learning
plot_model(et, 'manifold')

# Feature Importance
plot_model(et, 'feature')

# Model Hyperparameter
plot_model(et, 'parameter')

plot_model(tuned_et, 'parameter')

evaluate_model(tuned_et)

# Summary Plot
interpret_model(et)

# Correlation Plot
interpret_model(et, plot = 'correlation')

# Reason Plot at Observation Level
interpret_model(et, plot = 'reason', observation = 10)

