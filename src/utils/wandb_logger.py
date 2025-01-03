import wandb


class WandbLogger:
    def __init__(self, project_name):
        self.run = wandb.init(project=project_name)
        self.metrics = {}

    def log_batch(self, metric_name, value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def log_epoch(self):
        self.run.log(self.metrics, commit=False)
        self.metrics = {}

    def log_single(self, metric_name, value):
        self.run.log({metric_name: value})

    def finish(self):
        self.run.finish()
