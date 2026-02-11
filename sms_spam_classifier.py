import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
   
    transformed = [
        ps.stem(word) for word in tokens 
        if word.isalnum() and word not in stop_words and word not in punctuation
    ]

    return " ".join(transformed)

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title('📧 Email/SMS Spam Classifier')
st.markdown("Enter a message below to check if it's legitimate or spam.")

input_sms = st.text_area('Message Content:', placeholder="Type or paste your message here...", height=150)

if st.button('Analyze Message'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Display Result
        st.divider()
        if result == 1:
            st.error('🚨 **This is SPAM!**')
            st.info("Common spam markers found: Urgency, suspicious links, or reward-based language.")
        else:
            st.success(' **This is NOT Spam (Ham)**')
            st.balloons()

