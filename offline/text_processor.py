import spacy
import re
import unicodedata

class TextProcessor:
    """
    Utilize spaCy NLP model to process text
    """
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.document_filter = self.build_exclusion_pattern([
            'chapter',
            'topic'
        ])

        # Greek char -> Latin term mappings
        self.GREEK_MAP = {
            # Upper case
            'Α': 'alpha', 'Β': 'beta', 'Γ': 'gamma', '∆': 'delta', 'Ε': 'epsilon',
            'Ζ': 'zeta', 'Η': 'eta', 'Θ': 'theta', 'Ι': 'iota', 'Κ': 'kappa',
            'Λ': 'lambda', 'Μ': 'mu', 'Ν': 'nu', 'Ξ': 'xi', 'Ο': 'omicron',
            'Π': 'pi', 'Ρ': 'rho', 'Σ': 'sigma', 'Τ': 'tau', 'Υ': 'upsilon',
            'Φ': 'phi', 'Χ': 'chi', 'Ψ': 'psi', 'Ω': 'omega',

            # Lower case
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
            'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
            'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
            'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'ς': 'sigma', 'τ': 'tau',
            'υ': 'upsilon', 'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
        }
        self.GREEK_PATTERN = re.compile("|".join(map(re.escape, self.GREEK_MAP.keys())))

    def split_sentences(self, text: str):
        """
        Function to split a paragraph into valid sentences
        """
        doc = self.nlp(text)

        ret = []
        for sent in doc.sents:
            notrait_text = sent.text.strip()
            if notrait_text and not self.document_filter.search(notrait_text):
                ret.append(self.clean_text(notrait_text))
        
        return ret
    
    def lemmatize(self, text: str):
        """
        Function to reduce words inside a string into their base dictionary form
        """
        
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def clean_text(self, text):
        """
        Function to eliminate section strings
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
    
    def simplify_phrase(self, text: str):
        """
        Function to simplify a phrase into its core structure using dependency parsing
        """
        
        # Parse the phrase
        doc = self.nlp(text)
        
        # Retrieve the root action, fallback to original phrase if failed
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if not root:
            return text

        # Retrieve the root action's dependencies
        parts = [root]
        for child in root.children:
            if child.dep_ in {"prep", "agent", "prt"}:
                parts.append(child)

        parts = sorted(parts, key=lambda x: x.i)
        
        return " ".join(t.text for t in parts)

    def normalize_text(self, text: str) -> str:
        """
        Canonical text normalization:
        - Unicode normalize
        - Expand Greek character into corresponding Latin term
        - Remove spaces around hypens
        - Replace hypen separators with spaces
        - Normalize separators
        - Remove unwanted symbols
        - Collapse spaces
        - Lowercase + strip
        """

        text = unicodedata.normalize("NFKC", text)

        # Expand Greek
        for k, v in self.GREEK_MAP.items():
            text = text.replace(k, v)

        # Normalize spaced hyphens
        text = re.sub(r"\s*-\s*", "-", text)

        # Replace hyphen between letters with space
        text = re.sub(r"(?<=[a-zA-Z])-(?=[a-zA-Z])", " ", text)

        # Normalize other separators
        text = re.sub(r"[,+/]", " ", text)

        # Remove unwanted chars
        text = re.sub(r"[()\[\]\|]'", "", text)

        # Collapse spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()





