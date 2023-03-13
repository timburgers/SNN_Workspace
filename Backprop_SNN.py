from SNN_Izhickevich import Izhikevich_SNN
import torch
import numpy as np
from tqdm import trange
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F 

class Dataset_height_drone(Dataset):
	# This loads the data and converts it
	def __init__(self,input_signal):
		self.df = pd.read_csv(datafile,header=None)
		self.df_labels = pd.read_csv(datafile,usecols=[4], header=None)
		self.df = self.df.drop(columns=[2,3,4])
		self.dataset = torch.tensor(self.df.to_numpy()).float()
		self.labels = torch.tensor(self.df_labels.to_numpy()).reshape(-1,1).float()
	
	# This returns the total amount of samples in your Dataset
	def __len__(self):
		return len(self.dataset)
	
	# This returns given an index the i-th sample and label
	def __getitem__(self, idx):
		return self.dataset[idx],self.labels[idx]



def train_SNN(network, train_dataset, epochs, batch_size):

    # Initialize NN and optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
    network.train()
    SNN_layers = 1

    #initialize total loss over epoch arrays
    loss_train_epochs = np.array([])
    loss_validate_epochs = np.array([])

    # Start the training
    for ep in trange(epochs, desc="Training NN"):
        loss_train_cum = 0
        for ds in train_dataset :
            # for data in ds:
            # 	print("Dataset size = ", data)
            dl_train = DataLoader(ds,batch_size=batch_size)

            snn_states = torch.zeros(SNN_layers, network.hidden_neurons)

            # for data in dl_train:
            # 	print(data[0].shape)
            # 	break
            for batch_train,label_train in dl_train:					
                # print("Batch size = ", batch_train.size())
                # print("Label size = ", label_train.size())
                # Forward step
                optimizer.zero_grad()
                oupt,snn_states = network(batch_train,snn_states)
                

                # Calculate the loss
                loss_train = F.mse_loss(oupt, label_train)
                # print("loss of train data =",loss_train)
                
                # Back propagation and update the weights
                loss_train.backward()

                # for p in model.parameters():
                # 	print("loss = ",p.grad.norm())
                optimizer.step()

                loss_train_cum += loss_train.detach().numpy()
            #print("loss of train data cum =",loss_train_cum)
        # Validate the trained NN
        loss_val_cum = 0

        for ds in validate_ds:
            rnn_hidden = torch.zeros(rnn_num_layers, rnn_hidden_dim)
            dl_validate = DataLoader(ds,batch_size = BATCH_SIZE)
            for batch_validate,label_validate in dl_validate:
                output_val, _ = self(batch_validate,rnn_hidden)
                loss_val = F.mse_loss(output_val,label_validate)
                loss_val_cum +=loss_val.detach().numpy()
        
        loss_train_epochs = np.append(loss_train_epochs,loss_train_cum)
        loss_validate_epochs = np.append(loss_validate_epochs,loss_val_cum)

    # winsound.Beep(1500,500)

    eval()

    # Plot the loss for the training and validation set 
    epochs_arr = np.arange(0,EPOCHS)
    plt.title("Training and validation loss for all epochs")
    plt.xlabel("Epochs [-]")
    plt.ylabel("Loss [-]")
    plt.plot(epochs_arr,loss_train_epochs, label="Training data")
    plt.plot(epochs_arr,loss_validate_epochs, label= "Validation data")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.show()

    print("-----------Training finished ----------")