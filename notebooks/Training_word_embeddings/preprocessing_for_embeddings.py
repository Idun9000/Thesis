import re
import contractions
import spacy
nlp = spacy.load("en_core_web_sm")

def clean_story_for_embeddings(text):
    # Expand contractions
    text = contractions.fix(text)
    # Remove HTML and Markdown 
    text = re.sub(r'<[^>]+>', ' ', text) 
    text = re.sub(r'\[[^\]]*\]', ' ', text) 
    # Replace long dashes with space 
    text = re.sub(r'[—–]', ' ', text)
    
    # Process text with spaCy (sentence segmentation + NER)
    doc = nlp(text)

    # Merge hyphenated word spans before iterating
    with doc.retokenize() as retokenizer:
        i = 0
        while i < len(doc) - 2:
            if (doc[i].text.isalnum() and 
                doc[i+1].text == '-' and 
                doc[i+2].text.isalnum()):
                retokenizer.merge(doc[i:i+3])
                i += 3
            else:
                i += 1

    cleaned_sentences = []
    for sent in doc.sents:
        tokens = []
        for token in sent:
            # Skip punctuation and whitespace tokens
            if token.is_punct or token.is_space:
                continue
            # Keep only tokens with valid characters
            clean_text = re.sub(r"[^a-zA-Z0-9\-]+", '', token.text)
            if not clean_text:
                continue
            # Keep capitalization for PERSON entities, lowercase the rest
            if token.ent_type_ == "PERSON":
                tokens.append(clean_text)
            else:
                tokens.append(clean_text.lower())
        if tokens:
            cleaned_sentences.append(tokens)
    return cleaned_sentences
