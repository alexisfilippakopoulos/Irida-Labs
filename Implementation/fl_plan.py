from collections import OrderedDict


class FL_Plan:
    """
    The Federated Learning plan to be used by the participating clients.
    Is created using the FL_Strategy object and holds information usefull towards the clients.
    """
    def __init__(self, epochs: int, lr:float, loss:str, optimizer: str, batch_size: int, model_weights: OrderedDict, classifier_weights: OrderedDict):
        self.GLOBAL_TRAINING_ROUNDS = epochs
        self.LEARNING_RATE = lr
        self.CRITERION = loss
        self.OPTIMIZER = optimizer
        self.BATCH_SIZE = batch_size
        self.model_weights = model_weights
        self.classifier_weights = classifier_weights

    def __repr__(self) -> str:
        header = "FL Plan"
        content = (
            f"GLOBAL TRAINING ROUNDS: {self.GLOBAL_TRAINING_ROUNDS}\n"
            f"LEARNING RATE: {self.LEARNING_RATE}\n"
            f"BATCH SIZE: {self.BATCH_SIZE}\n"
            f"CRITERION: {self.CRITERION.upper()}\n"
            f"OPTIMIZER: {self.OPTIMIZER.upper()}\n"
            f"MODEL WEIGHTS: {sum(p.numel() for p in self.model_weights.values())} parameters\n"
            f"CLASSIFIER WEIGHTS: {sum(p.numel() for p in self.classifier_weights.values())} parameters"
        )
        line_length = max(len(header), len(content.split('\n')[0]))
        
        formatted_output = f"{header.center(line_length, '-')} \n{content}\n{'-'.center(line_length, '-')}"
        return formatted_output