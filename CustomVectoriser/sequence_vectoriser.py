"""Vectoriser to convert texts to integer sequences and vice versa."""

import functools
import operator
import re
import typing

import nltk
import numpy
import sklearn.feature_extraction
import spacy
import tensorflow

NLP = spacy.load("en_core_web_sm")
NLP.Defaults.stop_words.update(nltk.corpus.stopwords.words("english"))
NLP.Defaults.stop_words.update(
    sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)


def custom_preprocessor(text: str, custom_stopwords: typing.Set[str]) -> str:
    """Process cleaned and concatenated text fields before tokenisation.

    Parameters
    ----------
    text : str
        combined text after cleaning unusual characters
    custom_stopwords : typing.Set[str]
        user provided custom stopwords

    Returns
    -------
    str
        input text without the stopwords
    """
    for stopword in sorted(set(custom_stopwords)):
        text = re.sub(f"\\b{stopword}\\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def custom_tokenizer(text: str) -> typing.List[str]:
    """Tokenize the pre-processed inputs.

    Parameters
    ----------
    text : str
        content of a particular ticket

    Returns
    -------
    typing.List[str]
        lemmatised tokens to take into consideration for building vocabulary
    """
    tokens = NLP(text)
    useful_tokens = filter(
        lambda token: not (token.is_currency or token.is_punct or token.is_space
                           or token.is_stop), tokens)
    lemmas = [
        token.lemma_ if token.lemma_ != "-PRON-" else token.text
        for token in useful_tokens
    ]

    return lemmas


class SequenceVectoriser(sklearn.feature_extraction.text.CountVectorizer):
    """Create a custom vectoriser considering lemmatisation, stop words and punctuations.

    Parameters
    ----------
    custom_stopwords : typing.Set[str]
        user provided custom stopwords
    maximum_sequence_length : int, optional
        maximum sequence length allowed, by default None
    pad_type : str, optional
        strategy for padding, by default "post"
    truncation_type : str, optional
        strategy for truncating, by default "post"
    quantile : float, optional
        quantile of lengths of training documents to use, by default 0.75
    """

    def __init__(self,
                 custom_stopwords: typing.Set[str],
                 maximum_sequence_length: int = None,
                 pad_type: str = "post",
                 truncation_type: str = "post",
                 quantile: float = 1.0) -> None:
        # pylint: disable=too-many-arguments
        self.tokens_to_indices = {
            "<PAD>": 0,
            "<OOV>": 1,
            "<START>": 2,
            "<END>": 3
        }
        self.indices_to_tokens = {
            index: token for token, index in self.tokens_to_indices.items()
        }
        self.maximum_sequence_length = maximum_sequence_length
        self.pad_type = pad_type
        self.truncation_type = truncation_type
        self.quantile = quantile

        super().__init__(lowercase=False,
                         preprocessor=functools.partial(
                             custom_preprocessor,
                             custom_stopwords=custom_stopwords),
                         tokenizer=custom_tokenizer,
                         token_pattern=None,
                         dtype=numpy.int64)

    def fit(self, raw_documents: typing.List[str]) -> object:
        """Fit the vectoriser on supplied documents.

        Parameters
        ----------
        raw_documents : typing.List[str]
            concatenated text fields of training tickets after generic cleanup

        Returns
        -------
        object
            Fitted vectoriser
        """
        # pylint: disable=arguments-differ
        super().fit(raw_documents)

        for word, index in self.vocabulary_.items():
            self.tokens_to_indices[word] = index + 4
            self.indices_to_tokens[index + 4] = word

        if self.maximum_sequence_length is None:
            transformed_outputs = self.text_to_word_sequences(raw_documents)
            self.maximum_sequence_length = self.get_pad_length(
                transformed_outputs)

        return self

    def get_pad_length(
            self, transformed_outputs: typing.List[typing.List[int]]) -> int:
        """Determine the uniform length for padding and truncation.

        Parameters
        ----------
        transformed_outputs : typing.List[typing.List[int]]
            list of list of indices corresponding to words in concatenated text for each document

        Returns
        -------
        int
            uniform length for padding
        """
        document_lengths = numpy.fromiter(
            (len(transformed_output)
             for transformed_output in transformed_outputs),
            dtype=int)
        document_length_upper_quartile = numpy.quantile(document_lengths,
                                                        self.quantile)

        return int(document_length_upper_quartile)

    def inverse_transform(self,
                          indexed_sequences: numpy.ndarray) -> typing.List[str]:
        """Convert integer sequences to space separated tokens.

        Parameters
        ----------
        indexed_sequences : numpy.ndarray
            integer sequences corresponding to tokens in vocabulary

        Returns
        -------
        typing.List[str]
            texts corresponding to tokens in vocabulary
        """
        # pylint: disable=arguments-differ
        recovered_texts = []
        for indexed_sequence in indexed_sequences:
            recovered_tokens = filter(
                lambda token: token != "<PAD>",
                operator.itemgetter(*indexed_sequence)(self.indices_to_tokens))
            recovered_texts.append(" ".join(recovered_tokens))

        return recovered_texts

    def pad_sequences(
            self, transformed_outputs: typing.List[typing.List[int]]
    ) -> numpy.ndarray:
        """Convert word sequences of varying lengths to uniform length.

        Parameters
        ----------
        transformed_outputs : typing.List[typing.List[int]]
            list of list of indices corresponding to words in concatenated text for each document

        Returns
        -------
        numpy.ndarray
            indices corresponding to words post-padded and post-truncated
        """
        padded_outputs = tensorflow.keras.preprocessing.sequence.pad_sequences(
            transformed_outputs,
            maxlen=self.maximum_sequence_length,
            padding=self.pad_type,
            truncating=self.truncation_type,
            dtype=self.dtype)

        return padded_outputs

    def text_to_word_sequences(
            self,
            raw_documents: typing.List[str]) -> typing.List[typing.List[int]]:
        """Convert texts to integer sequences.

        Parameters
        ----------
        raw_documents : typing.List[str]
            concatenated text for the tickets to be transformed

        Returns
        -------
        typing.List[typing.List[int]]
            list of list of indices corresponding to words in concatenated text for each document
        """
        analyzer = self.build_analyzer()
        transformed_outputs = []
        for raw_document in raw_documents:
            transformed_output = [2] + [
                self.tokens_to_indices.get(feature, 1)
                for feature in analyzer(raw_document)
            ] + [3]
            transformed_outputs.append(transformed_output)

        return transformed_outputs

    def transform(self, raw_documents: typing.List[str]) -> numpy.ndarray:
        """Convert text to sequences of indices accoring to the position of words in vocabulary.

        Parameters
        ----------
        raw_documents : typing.List[str]
            concatenated text for the tickets to be transformed

        Returns
        -------
        numpy.ndarray
            sequence of indices corresponding to words in concatenated text
        """
        transformed_outputs = self.text_to_word_sequences(raw_documents)
        padded_outputs = self.pad_sequences(transformed_outputs)

        return padded_outputs
