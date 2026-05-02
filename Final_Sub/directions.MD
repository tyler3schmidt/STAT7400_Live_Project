# Directions
best_predict.py and best_catboost_pipeline.pkl correspond to the best model, while master_predict.py and super_learner_nnls_model.pkl correspond to the meta-learner. Both python files contain a predict() function whose input is a feature vector.

# New Data
The file test_data.csv contains twenty rows of data using the fixed class preprocessing pipeline of our new images. All extraneous rows have been removed, with the first column being area. For reference, the first 10 are vegetables and the remaining rows are fruits.