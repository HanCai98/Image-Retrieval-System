import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import utils
from transformers import DistilBertTokenizer


class SimpleTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.word2id = {}  # only contain known words
        self.num_words = 0  # does not include <PAD> and <UNKNOWN>.
        # <UNKNOWN> index: self.num_word; <PAD> index: self.num_word + 1
        self.max_len = 0

    def construct(self, texts, min_freq=5):
        '''
        :param texts: str, words are split by whitespace. list of list of word.
        :param min_freq: if the frequency of word is less than min_freq, it will be identified as <UNKNOWN>
        '''
        word_freq = {}
        for text in texts:
            if len(text) > self.max_len:
                self.max_len = len(text)
            tokens = [self.lemmatizer.lemmatize(w.lower()) for w in text]
            for word in tokens:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
        word2id = {}
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in word2id:
                word2id[word] = len(word2id)
        self.num_words = len(word2id)
        self.word2id = word2id

    def convert_to_id(self, sentence):
        '''
        :param sentence: str (split by whitespace)
        :return: list of int
        '''
        sentence = sentence.lower().split(' ')
        tokens = self.lemmatizer.lemmatize(sentence)
        ids = [self.num_words + 1] * self.max_len  # initialize with <PAD>
        for i, word in enumerate(tokens):
            if i >= self.max_len:
                break
            if word in self.word2id:
                ids[i] = self.word2id[word]
            else:
                ids[i] = self.num_words
        return ids


# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()
# text = 'A man with pierced ears is wearing glasses and an orange hat.'.lower()
# tokens = [lemmatizer.lemmatize(word) for word in text.split(' ')]
# print(tokens)


if __name__ == '__main__':
    texts = None
    dic = utils.load_json('data/train.json')
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)

