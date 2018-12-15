# coding: utf-8
import lyx
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.model_selection import ShuffleSplit
import model as md
torch.manual_seed(1)

EPOCH = 300
HIDDEN_DIM = 500
NUM_LAYERS = 6


START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix = {0: 0, 1: 1, START_TAG: 2, STOP_TAG: 3}


class Example():
    def __init__(self, text, feature, label):
        self.text = feature
        self.feature = feature
        self.label = label


def main():
    examples = lyx.io.load_pkl("data/train")
    examples = np.array(examples)

    EMBEDDING_DIM = np.array(examples[0].feature).shape[1]

    model = md.BiLSTM_CRF(tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)

    # Check predictions before training
    # with torch.no_grad():
    #     precheck_feature = torch.tensor(
    #         examples[10].feature, dtype=torch.float)
    #     result = model(precheck_feature)
    #     print(result)

    # Run training

    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          weight_decay=0, momentum=0.9)

    ss = ShuffleSplit(n_splits=EPOCH, test_size=0.025, random_state=0)

    epoch = 0
    for train_set, validation_set in ss.split(examples):

        epoch += 1
        print(epoch)

        # train_loss = 0
        for example in examples[train_set]:

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            feature = torch.tensor(example.feature, dtype=torch.float).cuda()
            label = torch.tensor(example.label, dtype=torch.long).cuda()

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(feature, label)
            # train_loss += loss

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()

            print("train_loss", loss)

        validation_loss = 0
        for example in examples[validation_set]:
            feature = torch.tensor(example.feature, dtype=torch.float).cuda()
            label = torch.tensor(example.label, dtype=torch.long).cuda()
            loss = model.neg_log_likelihood(feature, label)
            validation_loss += loss
        print("validation_loss", validation_loss/len(validation_set))

        if epoch % 50 == 0:
            torch.save(model.state_dict(), './model.pth')

    path = "xyx_model_training/test.pkl"
    examples_test = pickle.load(open(path, "rb"))

    test_loss = 0
    for example in examples_test:
        feature = torch.tensor(example.feature, dtype=torch.float).cuda()
        label = torch.tensor(example.label, dtype=torch.long).cuda()
        loss = model.neg_log_likelihood(feature, label)
        test_loss += loss
    print("test_loss", test_loss/len(examples_test))

    # Check predictions after training
    with torch.no_grad():
        precheck_feature = torch.tensor(
            examples_test[10].feature, dtype=torch.float).cuda()
        precheck_label = torch.tensor(
            examples_test[10].label, dtype=torch.long).cuda()
        result = model(precheck_feature)
        print("prediction", result)
        print("label", precheck_label)

    torch.save(model.state_dict(), './model.pth')

    model = torch.load('./model.pth')


if __name__ == "__main__":
    main()
