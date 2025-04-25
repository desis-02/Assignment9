# Assignment9
Purpose:
This project uses basic Natural Language Processing (NLP) techniques to analyze text files and infer topics and authorship based on linguistic features.

Design and Implementation:
- Tokenization, stemming, and lemmatization done with NLTK.
- Named Entity Recognition (NER) performed to detect important named concepts.
- Trigram modeling used to compare similarity of writing styles for authorship attribution.

Attributes and Methods:
- preprocess(text): Tokenizes and cleans text.
- stem_and_lemmatize(tokens): Returns stems and lemmas.
- ner(text): Extracts named entities.
- most_common_tokens(tokens): Lists most common words.
- generate_trigrams(tokens): Forms trigrams (groups of three consecutive words).
- compare_trigrams(base, target): Measures trigram overlap between two texts.

Limitations:
- Small dataset, results are heuristic.
- Trigram comparison assumes similar phrasing styles; large vocabulary differences reduce accuracy.
- Named Entity Recognition not fine-tuned for medieval fantasy or cosmic horror domains.
