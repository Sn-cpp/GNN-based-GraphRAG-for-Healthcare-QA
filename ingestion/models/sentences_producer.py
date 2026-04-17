import spacy
import re

class SentenceProducer:
    """
    Utilize spaCy NLP model to split text into sentences
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.document_filter = self.build_exclusion_pattern([
            'chapter',
            'topic'
        ])

    def split_sentences(self, text: str) -> list[str, ]:
        """
        Helper function to split a paragraph into valid sentences
        """
        doc = self.nlp(text)

        ret = []
        for sent in doc.sents:
            notrait_text = sent.text.strip()
            if notrait_text and not self.document_filter.search(notrait_text):
                ret.append(self.clean_text(notrait_text))
        
        return ret
    
    def clean_text(self, text):
        """
        Helper function to eliminate section strings
        """
        # Matches: I. OVERVIEW, II. STRUCTURE, III. ...
        section_pattern = r"\b[IVXLCDM]+\.\s+[A-Z][A-Z\s]+\b"
        return re.sub(section_pattern, "", text)
    
    def build_exclusion_pattern(self, keywords):
        """
        Builder function for words filtering
        """
        
        # escape keywords to avoid regex issues
        escaped = [re.escape(k) for k in keywords]
        pattern = r"\b(" + "|".join(escaped) + r")\b"
        return re.compile(pattern, re.IGNORECASE)