# Final-Year-University-Project
Class-conditioned Audio Diffusion from scratch

Prerequisites:

-   Download the GTZAN dataset, which can be found here https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    Place this in the "GTZAN_Genre_Collection" folder
    This is required to train the model and use the dataset.

-   Download PANNs model, which can be found here https://github.com/qiuqiangkong/audioset_tagging_cnn
    Place this in the "PANNs" folder
    This is required to provide the embeddings for the FAD score, which is used to evaluate the model
    You can download the specific model "Cnn14_mAP=0.431.pth" from here https://zenodo.org/records/3987831
    This model is used, since it matches the resolution of the 256x256 samples generated from the models