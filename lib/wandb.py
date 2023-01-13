import wandb

class WeightBiases():
    '''
        A script to initialize the object to connect to the Weight & Biases API.
    '''
    def __init__(self, dict_):
        wandb.config = dict_
        wandb.init(project="nlu_project", entity="simonebarattin", name=dict_["name"], config=wandb.config)
        self.wandb = wandb
        self.step = 0
    
    def log(self, dict_):
        self.wandb.log(dict_, step=self.step)
    
    def step_increment(self, increment):
        self.step += increment

    def finish(self):
        self.wandb.finish()

    def save(self, path: str):
        self.wandb.save(path)
