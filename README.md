# STAT7400_Live_Project

On your computer clone this repository then move into it, such that the parent directory is STAT7400_Live_Project

This requires having scikitlearn, pandas, and numpy downloaded. You can simply enter the virtual environment if thats easier through 
```bash
source .venv/bin/activate
```

Here simply run 
```bash
python predict.py x
```
with x being the covariates. This will return a prediction, either "fruit" or "vegetable"

The only necessary files are predict.py, scaler.pkl, and svm_model.pkl