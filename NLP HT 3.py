import nltk
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download('punkt_tab')
nltk.download('stopwords')

articles = """
Artificial intelligence is transforming healthcare and education systems.
AI and machine learning are the future of technology and innovation.
Healthcare systems are improving with artificial intelligence solutions.
Education technology is growing rapidly with AI based tools.
"""

# Lowercase
text = articles.lower()

print("Lowercase Text:")
print(text)

# Tokenization
tokens = word_tokenize(text)

print("\nTokens:")
print(tokens)

# Remove Stopwords and Punctuation
stop_words = set(stopwords.words('english'))

filtered = [word for word in tokens
            if word.isalpha() and word not in stop_words]

print("\nAfter Stopword Removal:")
print(filtered)

# Frequency Distribution
freq = FreqDist(filtered)

print("\nWord Frequency:")
for word,count in freq.items():
    print(word,":",count)

# Vocabulary
vocabulary = set(filtered)

print("\nVocabulary:")
print(vocabulary)

print("\nVocabulary Size:",len(vocabulary))

# Top 5 Words
print("\nTop 5 Frequent Words:")

for word,count in freq.most_common(5):
    print(word,":",count)

# Type Token Ratio
ttr = len(vocabulary)/len(filtered)

print("\nType Token Ratio:")
print(round(ttr,2))

# Dominant Topic
print("\nDominant Topic:")

print(freq.max())

# Bar Chart
freq.plot(10)

plt.title("Word Frequency Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()