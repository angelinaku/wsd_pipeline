from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from gensim.models import Word2Vec
from simple_elmo import ElmoModel

from torch import nn


def pad_sequences(
    sequences: List,
    maxlen: Optional[int],
    dtype: str = 'int32',
    padding: str = 'post',
    truncating: str = 'post',
    value: int = 0,
) -> np.array:
    """Pad sequences to the same length.
    from Keras

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. ' 'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = ()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" ' 'not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f'Shape of sample {trunc.shape[1:]} of sequence at position {idx}'
                f'is different from expected shape {sample_shape}'
            )

        if padding == 'post':
            x[idx, : len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x


def build_matrix(
    word_idx: Dict,
    embedding_path: str = '',
    embeddings_type: str = 'word2vec',
    max_features: int = 100000,
    embed_size: int = 300,
) -> Tuple[np.array, int, List]:
    """
    Create embedding matrix

    Args:
        embedding_path: path to embeddings
        embeddings_type: type of pretrained embeddings ('word2vec', 'glove'')
        word_idx: mapping from words to their indexes
        max_features: max features to use
        embed_size: size of embeddings

    Returns:
        embedding matrix, number of of words and the list of not found words
    """
    if embeddings_type not in ['word2vec', 'glove']:
        raise ValueError('Unacceptable embedding type.\nPermissible values: word2vec, glove')

    model = Word2Vec.load(embedding_path)

    # Creating Embedding Index
    embedding_index = {}
    for word in model.wv.vocab:
        coefs = np.asarray(model.wv[word])
        embedding_index[word] = coefs

    nb_words = min(max_features, len(word_idx))
    if embeddings_type in ['word2vec', 'glove']:
        embedding_size = embed_size if embed_size != 0 else len(list(embedding_index.values())[0])
        all_embs = np.stack(embedding_index.values())
        embed_mean, embed_std = all_embs.mean(), all_embs.std()

        if '<unk>' not in embedding_index:
            embedding_index['<unk>'] = np.random.normal(embed_mean, embed_std, (1, embedding_size))

        embedding_matrix = np.random.normal(embed_mean, embed_std, (nb_words + 1, embed_size))

        for word, num in word_idx.items():

            # possible variants of the word to be found in word to idx dictionary
            variants_of_word = [word, word.lower(), word.capitalize(), word.upper()]

            for variant in variants_of_word:

                embedding_vector = embedding_index.get(variant)
                if embedding_vector is not None:
                    embedding_matrix[num] = embedding_vector
                    break

        return embedding_matrix
    else:
        raise ValueError('Unacceptable embedding type.\nPermissible values: word2vec, glove')


class Embedder(nn.Module):
    """
    Transform tokens to embeddings
    """

    def __init__(self, word_to_idx: Dict, embeddings_path: str, embeddings_type: str, embeddings_dim: int = 0):
        super().__init__()
        self.weights_matrix = build_matrix(
            word_idx=word_to_idx, embedding_path=embeddings_path,
            embeddings_type=embeddings_type, max_features=len(word_to_idx), embed_size=embeddings_dim
        )
        self.weights_matrix = torch.tensor(self.weights_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(self.weights_matrix)
        self.embedding.weight.requires_grad = False

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        embed = self.embedding(x)
        return embed


class ELMo_Embedder(nn.Module):
    """
    Transform tokens to embeddings
    """

    def __init__(self, embeddings_path: str):
        super().__init__()
        self.model = ElmoModel()
        self.model.load(embeddings_path)
        self.sess = self.model.get_elmo_session()
        print('ELMo Embedding Model is Loaded')

    def forward(self, x: List) -> torch.Tensor:
        # embed = self.model.get_elmo_vectors(x)
        embed = self.model.get_elmo_vectors_session(x, self.sess)
        embed = torch.Tensor(embed)
        return embed
