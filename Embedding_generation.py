import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from faiss import IndexFlatL2
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define paths
dest_embed_dir = './Store2'
file_path = "./data/Ohio_revised_administrative_codes_data.xlsx"

# Load dataset
df = pd.read_excel(file_path)
selected_df = df.iloc[:10, :21]
selected_df = selected_df.dropna(how='all')

# Ensure required columns are of type string
columns_to_convert = ['text_body', 'part_1_name', 'name', 'masterTitle', 'org_1_name', 'datasetName']
for col in columns_to_convert:
    selected_df[col] = selected_df[col].astype(str)

print('Info of dataset:')
selected_df.info()
print('Columns of CSV:', selected_df.columns)

# Load locally downloaded embedding model
model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)

def generate_embeddings(texts, model, tokenizer, instruction="", max_length=4096):
    # Prepend instruction if specified
    formatted_texts = [f"{instruction}{text}" for text in texts]
    inputs = tokenizer(formatted_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean-pooling
    return F.normalize(embeddings, p=2, dim=1).cpu().numpy()

# Initialize metadata and text chunks
metadata_list = []
texts = []

text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=40)

for _, row in selected_df.iterrows():
    metadata = {
        'datasetName': row['datasetName'],
        'datasetId': row['datasetId'],
        'masterTitle': row['masterTitle'],
        'part_1_name': row['part_1_name'],
        'part_1_number': row['part_1_number'],
        'name': row['name'],
        'number': row['number'],
        'docTitle': row['docTitle'],
        'docNumber': row['docNumber'],
        'org_1_name': row['org_1_name'],
        'org_2_name': row['org_2_name'],
        'createdYear': row['createdYear'],
        'modifiedYear': row['modifiedYear'],
        'created_date': row['created_date'],
        'modified_date': row['modified_date'],
        'citationCount': row['citationCount'],
        'number_edits': row['number_edits'],
    }

    concatenated_text = row['text_body'].strip()
    chunks = text_splitter.split_text(concatenated_text)
    for chunk in chunks:
        location = f"Location of Regulation is : {row['datasetName']}"
        section = f"Section of Regulations : {row['name']}"
        category = f"Category of Regulations : {row['part_1_name']}"
        combined_text = f"{location} {section} {category} {chunk}"
        texts.append(combined_text.strip())
        metadata_list.append(metadata)

print(f"Added chunks: {len(texts)}")

# Generate embeddings for all texts
print("Generating embeddings...")
text_embeddings = generate_embeddings(texts, model, tokenizer)

# Check if the FAISS index already exists
if os.path.exists(dest_embed_dir):
    print("Loading existing FAISS index...")
    faiss_index = FAISS.load_local(folder_path=dest_embed_dir, embeddings=text_embeddings, allow_dangerous_deserialization=True)
    faiss_index.add_texts(texts, metadatas=metadata_list)
else:
    print("Creating new FAISS index...")
    faiss_index = IndexFlatL2(text_embeddings.shape[1])  # Initialize FAISS Index
    faiss_index.add(text_embeddings)  # Add vectors to the index

    # Save the FAISS index
    print("Saving FAISS index...")
    faiss_index.save_local(folder_path=dest_embed_dir)

print("Embedding and FAISS index creation completed.")
    
