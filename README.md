## ErgasiaDL

The chosen task for the Deep Learning project of MSc AI is "Musical Instrument Classification".

The [dataset](https://www.upf.edu/web/mtg/irmas) consists of several annotated instrument classes (11) and the developed approach was multiclass classification.

---

Folder structure:

- `irmasCNN`: contains the Python scripts related to the preprocessing and training of the augmented stereo Mel spectrogram dataset and the CNN architecture respectively
- `misc_architectures`: contains two tested architectures (CNN-LSTM combination and Autoencoders), not included in the final models
- `model_pt`: contains the trained model of the "irmasCNN" architecture
- `sample_wav`: contains a sample WAV audio file from the dataset used in the report notebook
- `transfer_learning`: contains the Python scripts related to the pre-trained VGG architecture

Lastly, the report of the training and evaluation process is also included