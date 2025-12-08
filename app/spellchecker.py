# app/spellchecker.py
import os
import string
from symspellpy import SymSpell, Verbosity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# naming convention: app/data/{lang}_word_frequency.txt, example: en_word_frequency.txt, hi_word_frequency.txt

DEFAULT_LANG = "en"


class SpellChecker:
    def __init__(self, max_edit_distance=2, prefix_length=7):
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        # Keep a cache of loaded SymSpell per language
        self._symspell_map = {}  # lang -> SymSpell instance

    def _freq_path_for_lang(self, lang: str):
        lang = (lang or DEFAULT_LANG).lower()
        fname = f"{lang}_word_frequency.txt"
        return os.path.join(DATA_DIR, fname)

    def _load_symspell_for_lang(self, lang: str):
       lang = (lang or DEFAULT_LANG).lower()

       # def _load_symspell_for_lang(self, lang: str):
       #    lang = (lang or DEFAULT_LANG).lower()

       if lang in self._symspell_map:
        return self._symspell_map[lang]
       #    if lang in self._symspell_map:
        #     return self._symspell_map[lang]

       freq_path = self._freq_path_for_lang(lang)
       #    freq_path = self._freq_path_for_lang(lang)

       if not os.path.exists(freq_path):
        self._symspell_map[lang] = None
        return None
       #    if not os.path.exists(freq_path):
       #     self._symspell_map[lang] = None
       #     return None

       sym = SymSpell(
        max_dictionary_edit_distance=self.max_edit_distance,
        prefix_length=self.prefix_length
       )
       #    sym = SymSpell(
       #     max_dictionary_edit_distance=self.max_edit_distance,
       #     prefix_length=self.prefix_length
       # )

       try:
        # Use load_dictionary for all languages (word<TAB>freq format)
         with open(freq_path, "r", encoding="utf-8") as f:
              ok = sym.load_dictionary(f, term_index=0, count_index=1)
       #  try:
       #     # Use load_dictionary for all languages (word<TAB>freq format)
       #     ok = sym.load_dictionary(freq_path, term_index=0, count_index=1)

       except Exception as e:
        print("SYMSPELL ERROR:", e)
        self._symspell_map[lang] = None
        return None
        #    except Exception as e:
        #     print("SYMSPELL ERROR:", e)
        #     self._symspell_map[lang] = None
        #     return None

       if not ok:
        print("LOAD returned FALSE for", lang)
        self._symspell_map[lang] = None
        return None
       #    if not ok:
       #     print("LOAD returned FALSE for", lang)
        #     self._symspell_map[lang] = None
       #     return None

       self._symspell_map[lang] = sym
       return sym

       #    self._symspell_map[lang] = sym
       #    return sym



    def correct_text(self, text: str, lang: str = DEFAULT_LANG, max_results=2) -> str:
        """
        Try to correct using language-specific dictionary. If no dictionary, return original text.
        Use lookup_compound to handle multi-word corrections.
        """
        if not text or not isinstance(text, str):
            return text
        sym = self._load_symspell_for_lang(lang)
        if sym is None:
            # no-op if we don't have a dictionary
            return text
        # use compound lookup for multi-word contexts
        suggestions = sym.lookup_compound(
            text, max_edit_distance=self.max_edit_distance
        )
        if suggestions:
            return suggestions[0].term
        return text
