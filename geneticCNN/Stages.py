import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from Stage import Stage


class Stages(torch.nn.Module):
    """
    This class config for stages of model

    # Parameters for init
    1. `*num_stages` : Tuple/int
        - The number of stages must be pass
        - If num_stages is int it will generate Tuple with lenght is num_stages and each element variable 3 => 10
        - Example: num_stages = 2 => num_stages = (3, 5)
    2. `gen` : Tuple (default None)
        - The binary code for stages follow the index of num_stages
    3. `input_chanel` : int (default 128)
        - Input chanel of Nodes
    4. `output_chanel` : int (default 128)
        - Output chanel of Nodes
    5. `kernel_size` : int (default 5)
        - Kernel size of Nodes

    # Returns
    - `Tensor`
        - Stages

    # Properties
    - `Stages.conv_inputs` : torch.nn.ModuleList
        - list of convs use for input
    - `Stages.stages` : torch.nn.ModuleList
        - list of stages
    - `Stages.conv_outputs` : torch.nn.ModuleList
        - list of convs use for output
    - `Stages.pools` : torch.nn.ModuleList
        - list of pools
    - `Stages.num_stages` : Tuple
        - list of number of nodes in stages
    - `Stages.gen` : Tuple
        - list of binary code of connections between nodes in stages
    - `Stages.gen_model` : dict
        - dictionary for encode of stages

    Raises
    ------
    - ValueError
        - type(num_stages) != int and type(num_stages) != tuple
        - gen is not None
            - type(gen) != tuple
            - len(gen) != len(self.num_stages:tuple)
    """

    def __init__(self, num_stages, gen=None, input_chanel=128, output_chanel=128, kernel_size=5):
        super(Stages, self).__init__()
        if type(num_stages) != int and type(num_stages) != tuple:
            raise ValueError("number of stages must be int/tuple")
        self.num_stages = num_stages if type(num_stages) == tuple else tuple(random.randint(3, 10) for _ in range(num_stages))

        if gen is not None:
            if type(gen) != tuple:
                raise ValueError("gen must be tuple")
            if len(gen) != len(self.num_stages):
                raise ValueError("lenght of gen not match with lenght of num_stages")
            self.stages = nn.ModuleList([Stage(num_nodes, binary_code, input_chanel=input_chanel, output_chanel=output_chanel, kernel_size=kernel_size)
                                        for num_nodes, binary_code in zip(self.num_stages, gen)])
        else:
            self.stages = nn.ModuleList([Stage(i, input_chanel=input_chanel, output_chanel=output_chanel, kernel_size=kernel_size) for i in self.num_stages])

        self.conv_inputs = nn.ModuleList()
        [self.conv_inputs.add_module('input_{}'.format(i), nn.Conv2d(input_chanel, output_chanel, kernel_size, padding="same")) for i in range(len(self.num_stages))]
        self.conv_outputs = nn.ModuleList()
        [self.conv_outputs.add_module('output_{}'.format(i), nn.Conv2d(input_chanel, output_chanel, kernel_size, padding="same")) for i in range(len(self.num_stages))]
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2, ceil_mode=True) for _ in range(len(self.num_stages))])
        self.gen = tuple(s.binary_code for s in self.stages)
        self.gen_model = {'S_{}'.format(i + 1): '-'.join(s.separated_connections) for i, s in enumerate(self.stages)}

    def __build_stages(self, x):
        for conv_input, stage, conv_output, pool in zip(self.conv_inputs, self.stages, self.conv_outputs, self.pools):
            x = F.relu(conv_input(x))
            x = stage(x)
            x = F.relu(conv_output(x))
            x = pool(x)
        return x

    def forward(self, x):
        return self.__build_stages(x)
