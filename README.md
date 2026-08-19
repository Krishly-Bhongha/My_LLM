# My_LLM

This project is an attempt to build a Large Language Model from scratch, starting from the fundamentals of text tokenization and working all the way up to training a Transformer-based language model. Below is the working of the repo:

1. Prepare the Dataset
Place your dataset inside:


The raw data can be a text or other supported data file, depending on how preprocess.py is implemented.

Example:


2. Configure Preprocessing
Open:


Update it according to your dataset. Check and modify:

Input file names and paths
File format handling
Text extraction logic
Cleaning and normalization
Output location
Any dataset-specific preprocessing rules
Run the script from the repository root:


Verify that the processed data is created in the location expected by the tokenizer pipeline.

3. Train the Tokenizer
Open:


This is the complete tokenizer training pipeline. It generally:

Loads the preprocessed dataset.
Builds or trains a tokenizer.
Creates the tokenizer vocabulary and required files.
Updates the tokenizer configuration in config.py.
Saves the trained tokenizer as a new folder inside:

Run it with:


After completion, confirm that a new tokenizer directory exists in Data/Tokenizer/. Keep its name and path available for the model configuration steps.

4. Create a New Transformer Model
Open:


Run:


The script asks for a model name and creates a new model directory:


The generated folder contains its own configuration and model files, such as:


Each model has an independent configuration, so changes made to one model do not automatically affect other models.

5. Configure the Model
Open the generated files:


Update the configuration as needed, including values such as:

Tokenizer path
Vocabulary size
Context or sequence length
Embedding size
Number of Transformer layers
Number of attention heads
Batch size
Learning rate
Training and checkpoint paths
Modify model.py only when changing the model architecture or its implementation.

Ensure that the model configuration matches the tokenizer created in the previous step.

6. Initialize Model Parameters
Run the generated model.py:


This initializes the model parameters and prepares the model for training. Confirm that the expected model files or checkpoints are created.

7. Train the Model
Open:


Select or configure the model you want to train. Make sure it points to:

The correct model directory
The correct tokenizer
The prepared dataset
The desired training settings
Start training with:


Training output and checkpoints should be saved according to the paths configured for that model.

8. Run Inference
After training is complete, go to:


Run:


The script asks for the model name. Enter the name of the model created under:


After the model loads, type messages to start chatting with it.