import nltk
from collections import Counter
nltk.download('punkt')


def mask_ngrams(text, n, mask_symbol):
    tokens = nltk.word_tokenize(text)
    ngrams = nltk.ngrams(tokens, n)
    freq_dist = Counter(ngrams)
    for gram, freq in freq_dist.items():
        if freq > 1:
            mask = [mask_symbol] * n
            text = text.replace(" ".join(gram), " ".join(mask))
    return text


if __name__ == '__main__':
    text = "The quick brown . The quick brown fox jumps over the lazy dog."
    masked_text = mask_ngrams(text, 2, "*")
    print(masked_text)

