import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr
# Load pre-trained model and tokenizer
def speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak something...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("you said :",text)
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand your speech.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
inputpp = speech()
model_name = 'bert-base-uncased'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Define emotion labels
emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
# Get user input
user_input = inputpp
# Tokenize and encode user input
tokens = tokenizer.encode_plus(user_input, padding='max_length', truncation=True,
                               max_length=128, return_tensors='pt')

# Make prediction
with torch.no_grad():
    model.eval()
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).squeeze()
    predicted_emotion_index = torch.argmax(probabilities).item()
    predicted_emotion = emotion_labels[predicted_emotion_index]

# Output the predicted emotion
#print("predicted emotion: ",predicted_emotion)
print(predicted_emotion)