from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model 1

# model_name = "facebook/bart-large-mnli"

# print("1")

# # Download and save the model and tokenizer locally.
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("2")

# # Save to a local folder, e.g., './my_local_model'
# model.save_pretrained("D:/Udemy/RAG_Pipeline_Project/my_project/models/zero_shot/Deep_Models")
# tokenizer.save_pretrained("D:/Udemy/RAG_Pipeline_Project/my_project/models/zero_shot/Deep_Models")

# print("3")

# embedding

from sentence_transformers import SentenceTransformer

# # Download the model.
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# print("1")
# # Save the model locally.
# model.save("D:/Udemy/RAG_Pipeline_Project/my_project/models/embeddings/Embedding_Models")
# print("2")

# # model 2
model_name = "unitary/toxic-bert"

print("1")

# Download and save the model and tokenizer locally.
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("2")

# Save to a local folder, e.g., './my_local_model'
model.save_pretrained("D:/Udemy/RAG_Pipeline_Project/my_project/models/toxicity/Unitary_Models")
tokenizer.save_pretrained("D:/Udemy/RAG_Pipeline_Project/my_project/models/toxicity/Unitary_Models")

print("3")

# embedding 2

# from sentence_transformers import SentenceTransformer

# # Download the model.
# model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# print("1")
# # Save the model locally.
# model.save("D:/Udemy/RAG_Pipeline_Project/my_project/models/Hallucination")
# print("2")

# model 2

# model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# print("1")

# # Download and save the model and tokenizer locally.
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# print("2")

# # Save to a local folder, e.g., './my_local_model'
# model.save_pretrained("D:/Udemy/RAG_Pipeline_Project/my_project/models/cross_encoder/Rerank")
# tokenizer.save_pretrained("D:/Udemy/RAG_Pipeline_Project/my_project/models/cross_encoder/Rerank")

# print("3")
