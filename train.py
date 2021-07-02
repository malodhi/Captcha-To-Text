from pathlib import Path
import datetime
import os, glob
import torch
import yaml
import numpy as np
import torch.optim as optim
from model import *


class Trainer(object):
    def __init__(self, config_file_path):
        super(Trainer, self).__init__()
        self.config_file_path = config_file_path
        self.param = self.get_parameters()
        model_class = self.param['model']['name']
        self.model = eval(model_class)()

        optim_class = self.param['optimizer']['name']
        optim_kwargs = self.param['optimizer']['kwargs']
        model_params = self.model.parameters()
        self.optimizer = eval(optim_class)(model_params, **optim_kwargs)
        self.criterion = eval(self.param['loss']['name'])()
        self.device = self.param['hyperparameters']['device']
        self.model_path = None
        
    def get_parameters(self):
        abs_file_path = Path(self.config_file_path).resolve().as_posix()
        with open(abs_file_path) as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def train(self, train_set, val_set):

        epochs = self.param['hyperparameters']['epochs']
        batch_size = self.param['hyperparameters']['batch_size']
        device = self.device

        x_train, x_val = torch.tensor(train_set[0], dtype = torch.double), torch.tensor(val_set[0], dtype = torch.double)
        y_train, y_val = torch.tensor(train_set[1], dtype = torch.double), torch.tensor(val_set[1], dtype = torch.double)
        self.model = self.model.double()
        train_set_size = train_set[0].shape[0]
        train_loops = int(len(x_train) / batch_size)
        valid_loops = int(len(x_val) / batch_size)

        for epoch in range(epochs):
            print("Epoch : ", epoch)
            training_loss = 0
            batch_start_index, batch_end_index = 0, batch_size

            for train_iteration in range(train_loops):

                x_batch, y_batch = x_train[batch_start_index:batch_end_index] , y_train[:, batch_start_index: batch_end_index]

                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                training_loss += loss

                batch_start_index += 4
                batch_end_index += 4

            print("epoch : ", epoch , "   loss" , training_loss / train_set_size)

            validation_loss = 0
            batch_start_index, batch_end_index = 0, batch_size

            with torch.no_grad():
                for _ in range(valid_loops):

                    x_batch , y_batch = x_val[batch_start_index:batch_end_index], y_val[:, batch_start_index: batch_end_index]
                    output = self.model(x_batch)
                    loss = self.criterion(output, y_batch)
                    validation_loss += loss

                    batch_start_index += 4
                    batch_end_index += 4

            if validation_loss < training_loss:
                snapshot_path = Path.joinpath(Path().parent.absolute() ,  "snapshot_model_epoch_" + str(epoch) + ".pt" )
                self.model_path = snapshot_path
                self.save_model()
                for f in glob.glob(os.getcwd() + '/snapshot_model_epoch_' + '*'):
                    print(f)
                    if f != snapshot_path.as_posix():
                        os.remove(f)

    def save_model(self):
        ppath = Path(self.model_path)
        if ppath.suffixes:
            dir_path = ppath.parent
        else:
            dir_path = ppath
        dir_path.mkdir(exist_ok=True, parents=True)

        torch.save(self.model, self.model_path)

    def load_model(self):
        self.model = torch.load(self.model_path)
        self.model.eval()

    def test(self, test_set):
        print(type(self.model_path))
        self.model = torch.load(self.model_path)
        test_set_size = len(test_set)
        x_test, y_test = torch.tensor(test_set[0], dtype = torch.double), torch.tensor(test_set[1], dtype = torch.double)

        test_loss = 0
        for test_example_no in range(test_set_size):
            x_test_example = x_test[test_example_no].reshape((1,3,32,32))
            y_test_example = y_test[:, test_example_no]
            output = self.model(x_test_example)
            test_loss  += self.criterion(output.reshape(-1), y_test_example)

        print("Test Loss:  " , test_loss / test_set_size)
        pass


def write_yaml(data, path):

    ppath = Path(path)
    if ppath.suffixes:
        dir_path = ppath.parent
    else:
        dir_path = ppath
    dir_path.mkdir(exist_ok=True, parents=True)

    with open(path, 'w') as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    config_params = {'model':       {'name': 'FirstNet'},
                     'optimizer' :  {'name' : 'optim.SGD',
                                    'kwargs': {'lr': 0.001, 'momentum': 0.9}
                                    },
                     'loss': {'name': 'nn.MSELoss'},
                     'hyperparameters': {'epochs': 100,
                                        'batch_size': 4,
                                        'device': 'gpu'}
                     }

    write_yaml(config_params, "model_config.yaml")

    trainer = Trainer(config_file_path=  "model_config.yaml")


    ############## Random Testing #################
    x_train = np.random.random((200, 3, 32,32))  # (batch_size, channel, H, W)
    y_train = np.random.random((1, 200))
    x_val = np.random.random((50, 3,32,32))
    y_val  = np.random.random((1 , 50))
    x_test = np.random.random((50, 3, 32, 32))
    y_test = np.random.random((1, 50))

    trainer.train((x_train, y_train), (x_val, y_val) )
    # trainer.test((x_test,y_test))


