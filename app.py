# import os
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel, BertPreTrainedModel
# from torch.optim import AdamW
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import torch.nn as nn
# import numpy as np
# from transformers import BertConfig

# # Device setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load data
# df = pd.read_csv('dataset.csv')
# texts = df['text'].tolist()
# labels = df[['anger', 'fear', 'joy', 'sadness', 'surprise']].values

# # Train/test split
# train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# print(tokenizer)

# class EmotionDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         encoding = self.tokenizer(
#             self.texts[idx],
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_len,
#             return_tensors='pt'
#         )
#         item = {key: val.squeeze(0) for key, val in encoding.items()}
#         item['labels'] = torch.FloatTensor(self.labels[idx])
#         return item

# train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
# test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16)

# class BertForMultiLabel(BertPreTrainedModel):
#     def __init__(self, config, num_labels=5):
#         super().__init__(config)
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.sigmoid = nn.Sigmoid()
#         self.init_weights()
    
#     def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
#         outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = self.dropout(outputs.pooler_output)
#         logits = self.classifier(pooled_output)
#         probs = self.sigmoid(logits)
#         loss = None
#         if labels is not None:
#             loss_fn = nn.BCELoss()
#             loss = loss_fn(probs, labels)
#         return {'loss': loss, 'logits': probs}

# config = BertConfig.from_pretrained('bert-base-uncased')
# model = BertForMultiLabel.from_pretrained('bert-base-uncased', config=config)
# model.to(device)

# model_path = "saved_model.pth"

# # Check if saved model exists, then load it; else train and save
# if os.path.exists(model_path):
#     print(f"Loading model from {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location=device))
# else:
#     print("Training model...")
#     optimizer = AdamW(model.parameters(), lr=2e-5)

#     for epoch in range(3):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs['loss']
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss / len(train_loader)
#         print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

#     # Save the trained model
#     torch.save(model.state_dict(), model_path)
#     print(f"Model saved to {model_path}")

# # Evaluation
# model.eval()

# all_preds = []
# all_true = []

# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].cpu().numpy()
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         probs = outputs['logits'].cpu().numpy()
#         preds = (probs >= 0.5).astype(int)
#         all_preds.append(preds)
#         all_true.append(labels)

# y_pred = np.vstack(all_preds)
# y_true = np.vstack(all_true)

# print(classification_report(y_true, y_pred, target_names=['anger', 'fear', 'joy', 'sadness', 'surprise']))

# # Streamlit app (assuming model, tokenizer, and device are ready as above)
# import streamlit as st

# emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# st.title("Emotion Prediction")

# # Input text box
# new_text = st.text_area("Enter text to analyze emotion:", value="Taking Wasif in my team was a shameful experience")

# if st.button("Predict Emotions"):
#     encoding = tokenizer(
#         new_text,
#         truncation=True,
#         padding='max_length',
#         max_length=128,
#         return_tensors='pt'
#     )
    
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
    
#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         probs = outputs['logits'].cpu().numpy()[0]
#         preds = (probs >= 0.5).astype(int)
    
#     predicted_emotions = [emotion_labels[i] for i, val in enumerate(preds) if val == 1]
    
#     if predicted_emotions:
#         st.success(f"Predicted emotions: {', '.join(predicted_emotions)}")
#     else:
#         st.info("No strong emotions detected.")



import os
import re
import pandas as pd
import numpy as np
import nltk
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, AdamW
from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download NLTK resources ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load Dataset ---
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('dataset.csv')
    return df

df = load_data()
texts = df['text'].tolist()
labels = df[['anger', 'fear', 'joy', 'sadness', 'surprise']].values
emotion_labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

# --- Setup device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======= Naive Bayes Model Setup =======

@st.cache_data(show_spinner=False)
def preprocess_texts(texts):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    corpus = []
    for sentence in texts:
        review = re.sub('[^a-zA-Z]', ' ', str(sentence))
        review = review.lower().split()
        review = [lemmatizer.lemmatize(word) for word in review if word not in stop_words]
        corpus.append(' '.join(review))
    return corpus

corpus = preprocess_texts(texts)

@st.cache_resource(show_spinner=False)
def train_naive_bayes(corpus, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    nb = MultinomialNB()
    multi_target_nb = MultiOutputClassifier(nb)
    multi_target_nb.fit(X_train, y_train)
    y_pred = multi_target_nb.predict(X_test)
    acc_per_label = {emotion_labels[i]: accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(len(emotion_labels))}
    report = classification_report(y_test, y_pred, target_names=emotion_labels, output_dict=True)
    return multi_target_nb, vectorizer, acc_per_label, report

multi_target_nb, vectorizer, nb_acc, nb_report = train_naive_bayes(corpus, labels)

# ======= BERT Model Setup =======

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.FloatTensor(self.labels[idx])
        return item

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class BertForMultiLabel(BertPreTrainedModel):
    def __init__(self, config, num_labels=5):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fn = nn.BCELoss()
            loss = loss_fn(probs, labels)
        return {'loss': loss, 'logits': probs}

config = BertConfig.from_pretrained('bert-base-uncased')
model = BertForMultiLabel.from_pretrained('bert-base-uncased', config=config)
model.to(device)

model_path = "saved_model.pth"

# Load or train BERT model
# @st.cache_resource(show_spinner=False)
# @st.cache_resource(show_spinner=False)
def train_or_load_bert_model(train_texts, train_labels, tokenizer, model_path, device):
    # Create Dataset & DataLoader inside cache function
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    config = BertConfig.from_pretrained('bert-base-uncased')
    model = BertForMultiLabel.from_pretrained('bert-base-uncased', config=config)
    model.to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        optimizer = AdamW(model.parameters(), lr=2e-5)
        model.train()
        for epoch in range(3):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return model

model = train_or_load_bert_model(model, train_loader, model_path, device)

# Evaluate BERT
def evaluate_bert(model, test_loader, device):
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs['logits'].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_true.append(labels)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)
    acc_per_label = {}
    for i, label in enumerate(emotion_labels):
        acc_per_label[label] = (y_true[:, i] == y_pred[:, i]).mean()
    report = classification_report(y_true, y_pred, target_names=emotion_labels, output_dict=True)
    return acc_per_label, report

bert_acc, bert_report = evaluate_bert(model, test_loader, device)

# --- Streamlit UI ---

st.title("Multi-label Emotion Classification Demo")

st.markdown("## Model Accuracies")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Naive Bayes Accuracy per Emotion")
    for emotion, acc in nb_acc.items():
        st.write(f"**{emotion.capitalize()}:** {acc:.3f}")

with col2:
    st.subheader("BERT Accuracy per Emotion")
    for emotion, acc in bert_acc.items():
        st.write(f"**{emotion.capitalize()}:** {acc:.3f}")

st.markdown("---")

st.markdown("## Enter Text to Predict Emotions")

user_input = st.text_area("Enter your text here:", value="Taking Wasif in my team was a shameful experience")

if st.button("Predict with Both Models"):
    # Naive Bayes prediction
    nb_processed = preprocess_texts([user_input])
    X_input = vectorizer.transform(nb_processed).toarray()
    nb_pred = multi_target_nb.predict(X_input)[0]
    nb_pred_labels = [emotion_labels[i] for i, val in enumerate(nb_pred) if val == 1]

    # BERT prediction
    encoding = tokenizer(
        user_input,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = outputs['logits'].cpu().numpy()[0]
        bert_pred = (probs >= 0.5).astype(int)
    bert_pred_labels = [emotion_labels[i] for i, val in enumerate(bert_pred) if val == 1]

    st.markdown("### Predictions")
    st.write(f"**Naive Bayes predicts:** {', '.join(nb_pred_labels) if nb_pred_labels else 'No strong emotions detected.'}")
    st.write(f"**BERT predicts:** {', '.join(bert_pred_labels) if bert_pred_labels else 'No strong emotions detected.'}")
