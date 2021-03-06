# If data is not present then skip
import os.path

import pytest
import torch


@pytest.mark.skipif(not os.path.exists('data/processed/train.pt'), reason="Train files not found")
@pytest.mark.skipif(not os.path.exists('data/processed/test.pt'), reason="Test files not found")

# See how much of data is run based on tests
# RUN test : coverage run -m pytest tests/
# See coverage : coverage report


class TestClass:
    def test_traindata_length(self):
        train_dataset = torch.load('data/processed/train.pt')
        N_train = 25000
        assert len(train_dataset) == N_train

    def test_two(self):
        test_dataset = torch.load('data/processed/test.pt')
        N_test = 5000
        assert len(test_dataset) == N_test

    def test_shape(self):
        train_dataset = torch.load('data/processed/train.pt')
        test_dataset = torch.load('data/processed/test.pt')

        for i, data in enumerate(train_dataset, 0):
            # get the inputs, batch size of 4
            images, _ = data
            assert images.shape == torch.Size([28,28])
        for i, data in enumerate(test_dataset, 0):
            # get the inputs, batch size of 4
            images, _ = data
            assert images.shape == torch.Size([28,28])

    def test_labels_represented(self):
        train_dataset = torch.load('data/processed/train.pt')
        test_dataset = torch.load('data/processed/test.pt')
        labels_train=[]
        labels_test=[]
        for _,data in enumerate(train_dataset,0):
            _ , lab = data
            # print(lab.item())
            labels_train.append(int(lab.item()))
        for _,data in enumerate(test_dataset,0):
            _ , lab = data
            # print(lab.item())
            labels_test.append(int(lab.item()))
        assert list(set(labels_train)) == [0,1,2,3,4,5,6,7,8,9]
        assert list(set(labels_test)) == [0,1,2,3,4,5,6,7,8,9]