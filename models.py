# models.py

from imp import init_frozen
from sentiment_data import *
from evaluator import *

from collections import Counter
import os
import numpy as np
import torch
from torch import nn, optim


######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, n_classes, vocab_size, emb_dim, n_hidden_units):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
       
        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        # PAD's embedding will not be trained and by default is initialized as zero
        # self.word_embeddings = torch.from_numpy(np.random.randn(self.vocab_size, self.emb_dim)).float() #out replace None with the correct implementation
        # self.word_embeddings[0,:] = torch.from_numpy(np.zeros((1,self.emb_dim))).float()
        self.word_embeddings = torch.empty(self.vocab_size, self.emb_dim)
        nn.init.xavier_normal_(self.word_embeddings)
        self.word_embeddings[0,:] = torch.from_numpy(np.zeros((self.emb_dim))).float()

        self.padding_idx = 0

        # TODO: implement the FFNN architecture
        # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
        # and the output class size using self.n_classes        
        
        self.emb = nn.Embedding.from_pretrained(embeddings = self.word_embeddings,
                                                freeze = False,
                                                padding_idx = self.padding_idx)

        self.h = nn.Sequential(nn.Linear(in_features = self.emb_dim, out_features = self.n_hidden_units, bias=True),
                                nn.ReLU())
        
        self.hout = nn.Linear(in_features = self.n_hidden_units,
                                out_features = self.n_classes,
                                bias = True)
        
        self.sm = nn.Softmax(dim=1) #check if the dimension is right

    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
        """
        # TODO: implement the forward function, which returns the logits
        
        # Lookup to the word embeddings
        x = self.emb(batch_inputs)
        # Compute av using sentence lengths
        x = torch.div(x.sum(dim=1), batch_lengths.reshape(-1,1))
        
        x = self.h(x)
        logits = self.hout(x)
        
        return logits
        # raise Exception("Not Implemented!")
        
    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function 
        # but should return a list of predicted labels
        
        logits = self.forward(batch_inputs, batch_lengths)
        pred_prob = nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(pred_prob, dim=1)

        return pred_class


##################################
# IMPLEMENT THE TRAINING METHODS #
##################################

def train_feedforward_neural_net(
    args,
    train_exs: List[SentimentExample], 
    dev_exs: List[SentimentExample]) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)

    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    vocab = [] # replace None with the correct implementation
    
    # Compute first the frequencies
    vocab_freq = {}
    for ex in train_exs:
        words, words_counts = np.unique(ex.words, return_counts=True)
        for w, f in zip(words, words_counts):
            if len(w)>1:
                vocab_freq[w] = vocab_freq[w] + f if w in list(vocab_freq.keys()) else f
    
    # Filter out those words with very low freq
    f=1 # -> the frequency tested were 1, 5, and 10. It was selected 5. Check report for more detail.
    vocab = list(np.array(list(vocab_freq.keys()))[np.where(np.array(list(vocab_freq.values()))>=f)])
    
    if args.glove_path is not None:
        f = open(args.glove_path)
        embedding_matrix = []
        #Add PAD
        embedding_matrix.append(np.repeat(0, args.emb_dim))
        #Add UNK
        embedding_matrix.append(np.random.rand(args.emb_dim))
        glove_vocab = []
        for line in f:
            if len(line.strip())>0:
                fields = line.split(' ')
                if fields[0] in vocab:
                    glove_vocab = glove_vocab + [fields[0]]
                    embedding_matrix.append([float(x) for x in fields[1:]])
        
        embedding_matrix = np.array(embedding_matrix)
        vocab = glove_vocab

    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next (as their indices have been hard-coded in several places)
    vocab = ["PAD", "UNK"] + vocab
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)

    # TODO: create the FFNN classifier
    # replace None with the correct implementation
    model = FeedForwardNeuralNetClassifier(n_classes=2, vocab_size=len(vocab), emb_dim=args.emb_dim, n_hidden_units = args.n_hidden_units)

    if args.glove_path is not None:
        model.emb.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float(), requires_grad = True)

    # TODO: define an Adam optimizer, using default config
    optimizer = torch.optim.Adam(model.parameters()) # replace None with the correct implementation
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) test with 1e-3 and 1e-2
    # optimizer = torch.optim.SGD(model.parameters(), lr=.2) # test with 0.1 and 0.2
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2) #-> test with 1e-3 and 1e-2
    
    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)

    # training
    best_epoch = -1
    best_acc = -1

    # adding CrossEntropy loss
    cross_entropy = nn.CrossEntropyLoss()
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh() # initiate a new iterator for this epoch

        model.train() # turn on the "training mode"
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch()
        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            # TODO: clean up the gradients for this batch
            optimizer.zero_grad()

            # TODO: call the model to get the logits
            # print(batch_inputs)
            logits = model(batch_inputs, batch_lengths)
            # print(logits)
            # TODO: calculate the loss (let's name it `loss`, so the follow-up lines could collect the stats)
            # cross_entropy was set just right before for-loop
            loss = cross_entropy(nn.functional.softmax(logits,dim=1), batch_labels)

            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)

            # TODO: backpropagation (backward and step)
            loss.backward()
            optimizer.step()

            # get another batch
            batch_data = batch_iterator.get_next_batch()

        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval() # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
            
            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))
    
    model.eval() # switch to the evaluation mode
    return model
