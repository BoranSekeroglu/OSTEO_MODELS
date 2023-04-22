# OSTEO_MODELS
UNIMODAL and MULTIMODAL CNN MODELS TO PREDICT OSTEOPOROSIS

UNIMODAL_CT.ipynb and UNIMODAL_MRI.ipynb are the saved models in Jupyter Notebook.
UNIMODAL_CT.html and UNIMODAL_MRI.html are the saved models in Jupyter Notebook as html.
trainable_unimodal.ipynb is the trainable unimodal model (training includes only two sample images in the file)

UNIMODALMR FOLDER: 

035516(119).jpg --> MRI Osteoporosis Patient Test Image, 
15812904_t1_tse_sag_7.jpg --> MRI Control Group Test Image, 
UNIMODAL.h5 --> Saved model and weights for MRI Unimodal Model, 
UNIMODALMR.py --> Python program for testing MRI images,
trainable.py --> trainable unimodal model example with 2 images training.


UNIMODALCT FOLDER: 

26246471.jpg --> CT Osteoporosis Patient Test Image, 
16204172.jpg --> MRI Control Group Test Image, 
UNIMODALCT.h5 --> Saved model and weights for CT Unimodal Model, 
UNIMODALCT.py --> Python program for testing CT images.


Note: The architectures of the UNIMODALCT.py and UNIMODALMR.py are same as in trainable_model.py.


Requirements:

Tensorflow Keras 2.11.0, Numpy 1.24.2
