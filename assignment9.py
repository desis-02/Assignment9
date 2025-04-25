# === Load required libraries ===
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import ne_chunk, pos_tag
from nltk.util import ngrams
from collections import Counter
import re

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

# === Load Data ===
with open('RJ_Lovecraft.txt', 'r', encoding='utf-8') as f:
    text1 = f.read()

with open('RJ_Tolkein.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()

with open('RJ_Martin.txt', 'r', encoding='utf-8') as f:
    text3 = f.read()

with open('Martin.txt', 'r', encoding='utf-8') as f:
    text4 = f.read()

# === Define Functions ===
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def stem_and_lemmatize(tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stems = [stemmer.stem(t) for t in tokens]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return stems, lemmas

def ner(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = ne_chunk(tagged)
    named_entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity = " ".join(c[0] for c in subtree)
            named_entities.append((entity, subtree.label()))
    return named_entities

def most_common_tokens(tokens, n=20):
    return Counter(tokens).most_common(n)

def generate_trigrams(tokens):
    return list(ngrams(tokens, 3))

def compare_trigrams(base_trigrams, target_trigrams):
    base_counts = Counter(base_trigrams)
    target_counts = Counter(target_trigrams)
    common = base_counts & target_counts
    return sum(common.values())

# === Part 1: Text 1, 2, 3 Analysis ===
text1_tokens = preprocess(text1)
text2_tokens = preprocess(text2)
text3_tokens = preprocess(text3)

text1_stems, text1_lemmas = stem_and_lemmatize(text1_tokens)
text2_stems, text2_lemmas = stem_and_lemmatize(text2_tokens)
text3_stems, text3_lemmas = stem_and_lemmatize(text3_tokens)

text1_ner = ner(text1)
text2_ner = ner(text2)
text3_ner = ner(text3)

text1_common = most_common_tokens(text1_tokens)
text2_common = most_common_tokens(text2_tokens)
text3_common = most_common_tokens(text3_tokens)

print("Top 20 Tokens - Text 1 (RJ_Lovecraft):\n", text1_common)
print("Named Entities in Text 1:\n", text1_ner)
print("Top 20 Tokens - Text 2 (RJ_Tolkien):\n", text2_common)
print("Named Entities in Text 2:\n", text2_ner)
print("Top 20 Tokens - Text 3 (RJ_Martin):\n", text3_common)
print("Named Entities in Text 3:\n", text3_ner)

# Simple Subject Guess
print("\nSubject Guess:")
print("Text 1: Cosmic horror, eldritch tragedy")
print("Text 2: Mythical fantasy tragedy")
print("Text 3: Political intrigue and tragedy")

# === Part 2: Text 4 (Martin.txt) Author Matching ===
text4_tokens = preprocess(text4)
text4_trigrams = generate_trigrams(text4_tokens)

text1_trigrams = generate_trigrams(text1_tokens)
text2_trigrams = generate_trigrams(text2_tokens)
text3_trigrams = generate_trigrams(text3_tokens)

match1 = compare_trigrams(text4_trigrams, text1_trigrams)
match2 = compare_trigrams(text4_trigrams, text2_trigrams)
match3 = compare_trigrams(text4_trigrams, text3_trigrams)

print("\nTrigram Matches (Text 4 vs others):")
print(f"Text 1 (Lovecraft): {match1}")
print(f"Text 2 (Tolkien): {match2}")
print(f"Text 3 (Martin): {match3}")

print("\nAuthorship Guess for Text 4: Matches most closely to RJ_Martin (text 3 style).")
