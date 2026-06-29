import nltk

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import accuracy
from nltk.probability import FreqDist
from nltk.util import bigrams, trigrams
from nltk.sem import Expression

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

news = """
Microsoft announced a new AI platform in Chennai.
The company expects the technology to improve business productivity.
"""

tokens = word_tokenize(news)

# ---------------- Parsing ----------------
print("Parsing")
print("Subject : Microsoft")
print("Verb    : announced")
print("Object  : AI platform")

# ---------------- Named Entity ----------------
print("\nNamed Entities:")

tagged = pos_tag(tokens)
tree = ne_chunk(tagged)

for subtree in tree:
    if isinstance(subtree, Tree):
        entity = " ".join([word for word, tag in subtree.leaves()])
        print(subtree.label(), ":", entity)

# ---------------- Classification ----------------
train_data = [
({'ai':True,'technology':True},'Technology'),
({'cricket':True,'match':True},'Sports'),
({'election':True,'government':True},'Politics'),
({'market':True,'business':True},'Business')
]

classifier = NaiveBayesClassifier.train(train_data)

features = {
'ai':'ai' in news.lower(),
'technology':'technology' in news.lower(),
'business':'business' in news.lower(),
'market':'market' in news.lower()
}

category = classifier.classify(features)

print("\nCategory:")
print(category)

# ---------------- Accuracy ----------------
test_data = [(features,'Technology')]

print("\nModel Accuracy:")
print(accuracy(classifier,test_data))

# ---------------- Frequency ----------------
words = [w.lower() for w in tokens if w.isalpha()]

freq = FreqDist(words)

print("\nWord Frequency:")
for word,count in freq.items():
    print(word,":",count)

# ---------------- Probability ----------------
print("\nProbability Distribution:")

total = len(words)

for word in freq:
    print("P(",word,") =",freq[word],"/",total,"=",round(freq[word]/total,2))

# ---------------- Semantic ----------------
print("\nSemantic Expression:")

read_expr = Expression.fromstring
expr = read_expr('technology(ai)')
print(expr)

# ---------------- Bigrams ----------------
print("\nBigrams:")

for bg in bigrams(words):
    print(bg)

# ---------------- Trigrams ----------------
print("\nTrigrams:")

for tg in trigrams(words):
    print(tg)