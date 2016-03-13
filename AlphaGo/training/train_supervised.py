import os, argparse

class supervised_trainer:
    def __init__(self,train_folder,model_folder):
        self.train_folder = train_folder
        self.model_folder = model_folder

    def train(self):
        # run CNNPolicy.create_network()
        # randomly select a tensor from the training folder
        # train
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("train_folder", help="Path to folder training samples")
    parser.add_argument("model_folder", help="Path to folder where the model params will be saved each epoch.")
    args = parser.parse_args()

    trainer = supervised_trainer(args.train_folder,args.model_folder)
    trainer.train()
