Link to Marsyas dataset for audio genre classification : http://marsyas.info/downloads/datasets.html

Use the script "Genre_classification_marsyas_generate_label_model_data.py" to create a json dump of mfcc coefficients for all files under each genre in the marsyas dataset
The output json is very large for entire dataset ~ 630MB and takes long to generate, so change the number of files selected in the above script to choose fewer files per genre