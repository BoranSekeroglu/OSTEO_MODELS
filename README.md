# OSTEO_MODELS
UNIMODAL and MULTIMODAL CNN MODELS TO PREDICT OSTEOPOROSIS

Multimodal.ipynb, UNIMODAL_CT.ipynb and UNIMODAL_MRI.ipynb are the saved models in Jupyter Notebook and can be used for TESTING OSTEOPOROSIS.

UNIMODAL_CT.html and UNIMODAL_MRI.html are the saved models in Jupyter Notebook as html.

trainable_unimodal.ipynb and trainable_multimodal.ipynb are the trainable models (training includes a few sample images)

UNIMODALMR FOLDER: 

035516(119).jpg --> MRI Osteoporosis Patient Test Image, 

15812904_t1_tse_sag_7.jpg --> MRI Control Group Test Image, 


UNIMODAL.h5 --> Saved final model and weights for MRI Unimodal Model, 

UNIMODALMR.py --> Python program for testing MRI images for Osteoporosis,

trainable.py --> trainable unimodal model example with 2 images training.


UNIMODALCT FOLDER: 

26246471.jpg --> CT Osteoporosis Patient Test Image, 

16204172.jpg --> MRI Control Group Test Image, 

UNIMODALCT.h5 --> Saved final model and weights for CT Unimodal Model, 

UNIMODALCT.py --> Python program for testing CT images for Osteoporosis.


MULTIMODAL FOLDER: 

mri0.jpg, mri1.jpg, ct0.jpg, ct1.jpg images are the same images as in unimodal folders.

multimodal.h5 --> Saved final model and weights for MRI multimodal Model, 

multimodal.py --> Python program for testing CT and MRI images for Osteoporosis as multimodal.

trainable_multimodal.py --> trainable multimoal model.

Note: The architectures of the UNIMODALCT.py and UNIMODALMR.py are same as in trainable_model.py, and the architecture of multimodal model includes two unimodal paths for CT and MRI images separately.


Requirements:

Tensorflow Keras 2.11.0, Numpy 1.24.2
