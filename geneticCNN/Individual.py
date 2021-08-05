import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from Stages import Stages


class Individual(torch.nn.Module):
    """
    This class config for single Individual

    # Parameters for init
    1. `num_stages` : Tuple/int (default None)
        - The number of stages will be random 2 => 10 if don't pass
    2. `gen` : Tuple (default None)
        - The binary code for stages follow the index of num_stages
    3. `input_size` : int (default 1)
        - input size will be receive
    4. `output_size` : int (default 10)
        - output size will be return, mean it is classes
    5. `input_chanel` : int (default 128)
        - Input chanel of Nodes
    6. `output_chanel` : int (default 128)
        - Output chanel of Nodes
    7. `kernel_size` : int (default 5)
        - Kernel size of Nodes

    # Returns
    - `Module`
        - Individual

    # Properties
    - `Individual.input` : Conv2d
        - where input received and format before pass to Stages
        - can be embedding for NLP
    - `Individual.Stages` : Stages
        - Stages
    - `Individual.output` : Sequential
        - where output will be return, contain fully connected layers
    - `Individual.output_size` : int
        - output size will be return, mean it is classes
    - `Individual.accuracy` : float
        - Accuracy of model need pass after train/test (is used like fitness)
    - `Individual.loss` : float
        - Loss of model need pass after train/test

    Raises
    ------
    - ValueError
    """

    def __init__(self, num_stages=None, gen=None, input_size=1, output_size=10, input_chanel=128, output_chanel=128, kernel_size=5):
        super(Individual, self).__init__()
        self.input = nn.Conv2d(input_size, input_chanel, 3, padding="same")
        self.Stages = Stages(random.randint(2, 10), input_chanel=input_chanel, output_chanel=output_chanel, kernel_size=kernel_size) if num_stages is None and gen is None else Stages(num_stages, gen, input_chanel=input_chanel, output_chanel=output_chanel, kernel_size=kernel_size)
        self.output = None
        self.output_size = output_size
        self.accuracy = None
        self.loss = None

    def build_output(self, size):
        return nn.Sequential(
            nn.Linear(size, 128),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(128, self.output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.input(x)
        x = self.Stages(x)
        x = torch.flatten(x, 1)
        if self.output is None:
            self.output = self.build_output(x.size()[1]).to(x.device)
        x = self.output(x)
        return x
