# Language Model project
Repository with the code for the NLU course project.

## Installation
First, install the required packages with a python's virtual environment:
```
$ python -m venv language_modeling
$ source language_modeling/bin/activate

(language_modeling) $ pip install --upgrade pip
(language_modeling) $ pip install -r requirements.txt
(language_modeling) $ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

## Running the training script
After installing the reauired packages, you can start training one of the language models with the following command:
```
(language_modeling) $ python main.py --baseline --cuda --wb
```
*Notice: here --baseline means the vanilla LSTM is going to be trained. The four models correspond to --baseline, --awd, --attention and --cnn*.

## Running the evaluation script
Once the desired model has been trained, this can be evaluated by running the following command:
```
(language_modeling) $ python evaluate/evaluate.py --weight models/model_weights/vanilla-lstm.pth --cuda
```
The same applies when testing the inference of the pre-trained model
```
(language_modeling) $ python evaluate/inference.py --weight models/model_weights/vanilla-lstm.pth --cuda
```

## Dataset stats
To extract the statistics reported in the report, please run the following command:
```
(language_modeling) $ python evaluate/corpora_stats.py
```
