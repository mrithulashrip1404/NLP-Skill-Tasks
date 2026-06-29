import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, FreqDist

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('omw-1.4')

resume_text = """
John Doe is a software engineer with 5 years of experience in Python,
Machine Learning, Data Analysis, SQL and Web Development.
He has worked on AI projects and developed scalable applications.
"""

# Tokenization
tokens = word_tokenize(resume_text)
print("Tokens:")
print(tokens)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word.lower() for word in tokens
                   if word.isalnum() and word.lower() not in stop_words]

print("\nAfter Stopword Removal:")
print(filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]

print("\nStemmed Words:")
print(stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

print("\nLemmatized Words:")
print(lemmatized_words)

# POS Tagging
pos_tags = pos_tag(lemmatized_words)

print("\nPOS Tags:")
print(pos_tags)

# Frequency Distribution
freq = FreqDist(lemmatized_words)

print("\nFrequency Distribution:")
for word, count in freq.items():
    print(word, ":", count)