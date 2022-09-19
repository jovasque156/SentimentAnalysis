from nltk.corpus import stopwords
import numpy as np
import re
import argparse

# I created this function to preprocess the text before is tokenized.
def preprocess_text(text: str, symbols: str) -> str:
    """
    Preprocess the text removing special symbols, stop words, and alone letters
    :param text: a string of words to preprocess
    """
    # Separating symbols
    to_separate = ",.!?-_;:"
    for symbol in to_separate:
        text = text.replace(to_separate, ' '+symbol+' ')
    
    # Removing symbols
    for symbol in symbols:
        text = text.replace(symbol, ' ')
    
    # Removing double spaces
    text = re.sub(' +', ' ', text)

    return text

def remove_stop_words(words: list) -> list:
    """
    Remove stop words from the list
    :param words: list of words to perprocess
    """
    filtered = []
    for w in words:
        filtered = filtered + [w] if w not in stopwords.words('english') else filtered
    return filtered

def read_file(infile: str, outfile: str, symbols: str, stopping_words: bool):
    """
    Read file from infile path and return a processed one.
    :param infile: path of the that will be read
    :param outfile: path where new file will be saved
    :param symbols: set of symbols to be used in the preprocessing
    :param stopping_words: boolean indicating if stopping words are removed
    """
    f = open(infile)
    exs = []
    labels = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[0] else 1
                sent = " ".join(fields[1:])
            else:
                label = 0 if "0" in fields[0] else 1
                sent = fields[1]

            sent = sent.lower() # lowercasing
            
            sent = preprocess_text(sent, symbols)

            tokenized_cleaned_sent = remove_stop_words(list(filter(lambda x: x != '', sent.rstrip().split(" ")))) if stopping_words else list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            
            if len(tokenized_cleaned_sent)>0:
                exs.append(tokenized_cleaned_sent)
                labels.append(label)
    f.close()

    o = open(outfile, 'w')
    for ex, label in zip(exs, labels):
        o.write(str(label) + '\t' + " ".join([word for word in ex]) +' \n')
    o.close()

def read_blind_sst_file(infile: str, outfile:str, symbols: str, stopping_words: bool):
    """
    Read file from infile path and return a processed one.
    :param infile: path of the that will be read
    :param outfile: path where new file will be saved
    :param symbols: set of symbols to be used in the preprocessing
    :param stopping_words: boolean indicating if stopping words are removed
    """
    f = open(infile, encoding='utf-8')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            words = preprocess_text(line.lower(), symbols).split(" ")
            words = remove_stop_words(words) if stopping_words else words
            exs.append(words)
    f.close()

    o = open(outfile, 'w')
    for ex in exs:
        o.write(" ".join([word for word in ex]))
    o.close()

    return exs

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--train_path', type=str, default='data/train.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/test-blind.txt', help='path to blind test set (you should not need to modify)')
    parser.add_argument('--symbols', type=str, default="\/()<>`@#$%^&*[]{}=+1234567890", help='set of symbols to remove')
    parser.add_argument('--stopping_words', type=str, default="True", help='True if it is wanted to remove stopping words')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    # Load train, dev, and test exs and index the words.
    read_file(args.train_path, 'data/train_prepro.txt', args.symbols, args.stopping_words=='True')
    read_file(args.dev_path, 'data/dev_prepro.txt', args.symbols, args.stopping_words=='True')
    read_blind_sst_file(args.blind_test_path, 'data/test-blind_prepro.txt', args.symbols, args.stopping_words=='True')
    