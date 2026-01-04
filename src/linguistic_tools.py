import pandas as pd
from pathlib import Path
from functools import lru_cache


class PanLexTool:
    """Query PanLex dictionary for word validation"""

    def __init__(self, panlex_path: Path = Path("panlex.csv")):
        self.panlex_path = panlex_path
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy load PanLex data"""
        if self._df is None:
            self._df = pd.read_csv(self.panlex_path, sep="\t", dtype=str)
        return self._df

    def query(
            self,
            word: str = None,
            iso_639_3: str = None,
            language_name: str = None,
            max_size: int = 30
    ) -> str:
        """
        Query PanLex for matching entries

        Returns:
            Markdown table of results
        """
        df = self.df
        query = df

        if iso_639_3:
            query = query[query['639-3'].str.lower() == iso_639_3.lower()]

        if language_name:
            query = query[query['english name/var'].str.lower() == language_name.lower()]

        if word:
            query = query[query['vocab'].str.contains(word, case=False, na=False, regex=True)]

        if max_size:
            query = query.head(max_size)

        return query.to_markdown(index=False)

    def validate_word(self, word: str, lang_code: str) -> bool:
        """Check if word exists in dictionary for given language"""
        df = self.df
        query = df

        if lang_code:
            query = query[query['639-3'].str.lower() == lang_code.lower()]

        # Exact match (case-insensitive)
        query = query[query['vocab'].str.lower() == word.lower()]

        return len(query) > 0  # Returns True only if exact match found


import uroman as ur


class UromanTool:
    """Romanize text from any script"""

    def __init__(self):
        self._uroman = None

    @property
    def uroman(self):
        if self._uroman is None:
            self._uroman = ur.Uroman()
        return self._uroman

    def romanize(self, text: str) -> str:
        """Romanize text (e.g., Devanagari → Latin)"""
        return self.uroman.romanize_string(text)
