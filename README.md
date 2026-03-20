# STAT7400_Live_Project

On your computer clone this repository then move into it, such that the parent directory is STAT7400_Live_Project

This requires having scikitlearn, pandas, and numpy downloaded. You can simply enter the virtual environment if thats easier through 
```bash
source .venv/bin/activate
```

Here simply run 
```bash
python predict.py calibration_images/ test.csv
```
and this will save the predictions to test.csv, with columns image_id and prediction. 

To get the images inside, if on mac or linux enter this repository use pwd to get the working directory and copy it. Then move to where you folder of test images and run 
```bash
cp calibration_images /users/user/STAT7400_Live_Project
```
or the equivalent folder name and directory location.