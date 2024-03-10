# Chatbot for COVID-19 Education based on CORD-19 dataset

To run this code, you will need to download data from Kaggle ([link](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge/datahttps://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge/data)).
Put the 'archive.zip' file in the root of the repo. 

Then, run this code to create Chroma instance and populate it with documents:
```bash
python src/rag.py
```

Set your huggingface auth token to be able to access Llama-2 model: 
```bash
export HF_AUTH_TOKEN="your-token-here"
```

Run the demo application with:
```bash
python app.py
```
