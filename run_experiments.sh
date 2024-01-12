#!/bin/bash

python src/models/catboost/main_catboost_conformal.py --timeout 3600
python src/models/catboost/main_catboost_quantile.py --timeout 3600
python src/models/lightgbm/main_lightgbm_conformal.py --timeout 3600
python src/models/lightgbm/main_lightgbm_quantile.py --timeout 3600
python src/models/xgboost/main_xgboost_conformal.py --timeout 3600
python src/models/xgboost/main_xgboost_quantile.py --timeout 3600
python src/models/random_forest/main_rf_conformal.py --timeout 3600
python src/models/random_forest/main_rf_quantile.py --timeout 3600
python src/models/neural_networks/main_nn_bayesian.py --timeout 3600
python src/models/neural_networks/main_nn_conformal.py --timeout 3600
python src/models/neural_networks/main_nn_direct.py --timeout 3600
