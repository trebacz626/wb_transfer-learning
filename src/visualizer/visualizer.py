import wandb
from util import util

class Visualizer:
    def __init__(self, experiment):
        self.counter = 0
        self.experiment = experiment

    def log_images(self, visuals):
        self.counter+=1
        columns = [key for key, _ in visuals.items()]
        columns.insert(0,'epoch')
        result_table = wandb.Table(columns=columns)
        table_row = [self.counter]
        ims_dict = {}
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            wandb_image = wandb.Image(image_numpy)
            table_row.append(wandb_image)
            ims_dict[label] = wandb_image
        self.experiment.log(ims_dict)
        result_table.add_data(*table_row)
        self.experiment.log({"Result": result_table})
