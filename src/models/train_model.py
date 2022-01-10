import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
# from '../../data/processed' import '../../data/processed/train.pt';
from src.models.model import MyAwesomeModel
from torch.utils.data import DataLoader, TensorDataset
import hydra
import os

"""
DOCKER: Make the code reproducable 
1. Create docker file
2. Crete image: docker build -f trainer.dockerfile . -t trainer:latest
3. Run image and create container. When running the new created files are saved in the container and not on your laptop
4. To get the new files run: docker cp
"""

@hydra.main(config_name= "training_conf.yaml" ,config_path="../../conf")
def main(cfg):
    """Training loop
    """
    os.chdir(hydra.utils.get_original_cwd())
    print("Working directory : {}".format(os.getcwd()))
    print("Training day and night")    
    model = MyAwesomeModel() # My model
    criterion = torch.nn.CrossEntropyLoss()  # COst function
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr)  # Optimizer


    # Magic
    # wandb.watch(model, log_freq=cfg.print_every)


    train_dataset = torch.load(cfg.train_data)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCHSIZE,shuffle=True, num_workers=2)

    train_dataset = torch.load(cfg.test_data)
    testloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.BATCHSIZE,shuffle=True, num_workers=2)


    steps = 0
    running_loss = 0
    losses = []
    timestamp = []
    epochs = cfg.epochs
    print_every = cfg.print_every
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs, batch size of 4
            images, labels = data
            

            steps += 1

            optimizer.zero_grad()

            # Recontruct labels
            lab = torch.zeros(4, 10)
            for i in range(0,len(labels)):
                lab[i,int(labels[i])] = 1
            
            
            images = images[:, None, :, :]
            if images.shape[1] != 1 or images.shape[2] != 28 or images.shape[3] != 28:
                raise ValueError('Expected each sample to have shape [1, 28, 28]')
                
            output = model(images)
            print(output)
            print(output[0].shape)

            loss = criterion(output[0], lab)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:

                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)


                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                losses.append(running_loss / print_every)
                timestamp.append(steps)
                running_loss = 0
                # Make sure dropout and grads are on for training
                model.train()
    plt.plot(timestamp, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.savefig("reports/figures/training.png")

    #plt.show()
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, "src/models/checkpoint.pth")


    
def validation(model, testloader, criterion):
    """Model validation
    """
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images[:, None, :, :]
        # print(images.size())
        output = model.forward(images)[0]
        
        # Recontruct labels
        lab = torch.zeros(4, 10)
        for i in range(0,len(labels)):
            lab[i,int(labels[i])] = 1

        test_loss += criterion(output, lab).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # wandb.init(config= args)
    main()
