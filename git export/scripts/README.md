# land_cover_classification project

**project overview**
project overview:
this project performs supervised classification of high resolution orthophotos into land cover classes
it uses patch based feature extraction (RGB, Haralick, entropy), train/validation/test splits and scikit learn models
outputs include classified rasters, shapefiles, and performance metrics (csv).


**project structure**
main.py 				        # CLI. entry point for running train\prediction mode. run after completing subset_raster.py procedure
config.py 				        # global paths, constants,parameters adjustments											
data_utils.py 			    	# loads geotiff image, preprocess filters,masks, generates image patches							
rasterizer.py 			    	# rasterizes the labled shpfile and creates alligment with orthophoto raster				
evaluation_metrics.py 			# evaluation mode process. plots confusion matrix, calculates mterics		
feature_extractor.py 			# computes and extracts selected features from each proccesed patch							
model_trainer.py 			    # main trainning process file. trains randomforest,model evaluation,exports csv report, confusion matrix,
                                computes ground truth file from manually labeld shp file (later used in subset_raster.py),  landcover prediction model,exports evaluation reports	
predict_landcover.py 			# runs the land cover prediction model by loading trained classification model and predicts the land cover	class on new geoTIFF image based on patches output: predictions.csv,summarry analysis.csv with extracted top patch features		
shapefile_exporter.py 			# exports predicted patches as shpfile					
GUI.py 					        # project GUI structure									

**pre processing, outside of main.py scope, auxillaries (used before training):**

1.appendix_mask_file_config.py		# reconfigures standard class labels of shp files(landcover) for user desired labels.                                  
2.appendix_remap_mask_classes.py	# used after 'appendix_mask_file_config' and prior to model training. matches classification of classes between mask files and project classes        
3.subset_raster.py 			        # helper file for evaluation mode. creates a perfectly matching section of the input raster and ground truth raster for test running of the evaltuion mode. (creates subset_raster and sunset_ground truth TIFF files) 

**GUI based workflow**
1.appendix_mask_file_config.py
2.appendix_remap_mask_classes.py
3.**training** main.py -> model training   			##  input geotiff = subset_new_raster.epsg6991 ( for israel grid)
4. * optional- subset_raster.py                 # to create test data for evaluation	 
5.**evaluation** main.py -> evaluate mode				##  input geotiff = subset_new_raster.epsg6991, ground truth = subset_new_ground_truth.epsg6991  (for israel grid)

**vital files (rasters, subsets,shapefiles,weight_models etc)**

result.tif					                    # prediction model file (original orthophoto of aoi)
training_labels_rasterized.tif		            # prediction model file,ground truth raster, manually labeld (rasterized ground truth mask)
result_SUBSET.epsg6991.tif		             	# evaluation model test file, product of subset_raster.py (input raster subset)
training_labels_rasterized_SUBSET.epsg6991.tif	# evaluation model test file, product of subset_raster.py (ground truth mask subset)
randomforest_6_features.joblib			        # random forest weight model with 6 figures

**final outputs**

* **training outputs (model weight):**
    * randomforest_6_features.joblib				# exported trained model. saved by model_trainer.py
* **prediction outputs (predict_landcover.py):**
    * confidence_map_result.tif         # GIS heatmap of classification based on confidence result
    * landcover.shp                     # shp file containting all processed patches of the orthophoto	
    * randomforest_6_features_predictions.csv     # detailed per patch result table (with class confidence distribution) 	
	* randomforest_6_features_analysis.csv				# overall class prediction and metric anlysis							
    * randomforest_summary.csv          # per class summary of prediction model
* **evaluation outputs (model_trainer.py evaluation):**
    * randomforest_confusion_matrix_.csv # confusion matrix.png
    * randomforest_evaluation_report_.csv # analysis csv (precision, recall, F1 Score).
    * randomforest_confusion_matrix_.png # heatmap of the Confusion Matrix (for quick viewing).


**workflow**							
1. data preparation 
- place orthophoto + labeled shapefile in 'orthophoto/' folder
- rasterize the labeled shapefile mask (rasterizer.py)
2.train model
- run model_trainer.py → saves model in model_weights/
3.predict
- run predict_landcover.py → outputs classified raster and confidence map
4.export
- run shapefile_exporter.py → vectorize predictions to .shp

								
**dependencies:**
- python ≥ 3.10
libraries:
- numpy, pandas, os, sys, csv, datetime, math, subprocess, argparse, logging, tqdm, joblib, shutil, tkinter, matplotlib, seaborn
- geospatial and image processing: rasterio, geopandas, shapefile, shapely, cv2, scikit-image (skimage)
- ML and statistics: sklearn, scipy



####
this project is a proof of concept for land cover classification in outdoor areas,
it can be improved drastically both in efficiency (running time) and capabilities by upgrading the learning algorithm from random forest to convolutional neural network in order to overcome the segments (grid) constraints and apply better patterns recgonition in higher resolution.
