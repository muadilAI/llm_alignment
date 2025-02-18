## Instructions for Running the Optimization Models

The following code snippets should be executed based on the optimization method to be used on your approved Kaggle account. Before running the code, please ensure the following steps are completed:

### Prerequisites
1. **Add Tokens**: In order to use Huggingface and WanDB structures, you must add the token values obtained from their websites via the "Add Secret" button.
2. **GPU Setup**: A **GPUxT4** unit must be used for training, and from the "Environment Preferences" options, select "Pin to original environment".

### Code for Different Optimization Methods
- **For PPO Algorithm**: Run `llama_ppo_trainer.ipynb`.
- **For DPO Algorithm**: Run `dpo_trainer.ipynb`.
- **For KTO Algorithm**: Run `kto_trainer.ipynb`.

### After Model Training: Get the Success Metrics Scores
Once the models are trained, you can obtain the success metrics scores by running the `save-model-outputs.ipynb` script. Before running the code, make sure:
- The Huggingface and WanDB tokens are added via the "Add Secret" button.
- Specify the names of the model(s) for which you want to retrieve the test outputs.

### AI Feedback and Model Outputs
To obtain AI feedback for the models, run the `ai_feedback.ipynb` code. Before running, ensure:
- Huggingface, WanDB, and Together AI tokens are added via the "Add Secret" button.
- Provide a Google Drive link to save the output to Excel.
- Specify the names of the model(s) you want to test.
