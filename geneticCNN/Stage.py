import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class Stage(torch.nn.Module):
    """
    This class config for single stage of model

    # Parameters for init
    1. `*num_nodes` : int
        - The number of nodes must be pass
        - The number of nodes must be >= 2 
    2. `binary_code` : str (default None)
        - The binary code of connections between nodes
        - If don't pass, it will be create random
        - Only contain '0' and '1' in string
        - The string needs at least 1 bit '1'
    3. `input_chanel` : int (default 128)
        - Input chanel of Nodes
    4. `output_chanel` : int (default 128)
        - Output chanel of Nodes
    5. `kernel_size` : int (default 5)
        - Kernel size of Nodes

    # Returns
    - `Tensor`
        - Stage

    # Properties
    - `Stage.convs` : torch.nn.ModuleList
        - list of nodes (node is torch.nn.Conv2d)
    - `Stage.num_nodes` : int
        - number of nodes
    - `Stage.binary_code` : str
        - The binary code of connections between nodes
    - `Stage.inputs` : list
        - inputs connections between nodes
    - `Stage.outputs` : list
        - outputs connections between nodes
    - `Stage.separated_connections` : list
        - split binary code connections to encode of nodes

    Raises
    ------
    - ValueError
        - type(num_nodes) != int
        - num_nodes < 2
        - binary_code is not None:
            - type(binary_code) != str
            - not set(binary_code).issubset(set('10'))
            - len(binary_code) != int(num_nodes * (num_nodes - 1) / 2)
            - set(binary_code).issubset(set('0'))
    """

    def __init__(self, num_nodes, binary_code=None, input_chanel=128, output_chanel=128, kernel_size=5):
        super(Stage, self).__init__()
        if type(num_nodes) != int:
            raise ValueError("number of nodes must be int")
        if num_nodes < 2:
            raise ValueError("number of nodes must be >= 2")
        if binary_code is not None:
            if type(binary_code) != str:
                raise ValueError("binary code must be string")
            if not set(binary_code).issubset(set('10')):
                raise ValueError("binary code only contain '0' and '1' in string")
            if len(binary_code) != int(num_nodes * (num_nodes - 1) / 2):
                raise ValueError("binary code and number of nodes is not match")
            if set(binary_code).issubset(set('0')):
                raise ValueError("binary code needs at least 1 bit '1'")

        self.convs = nn.ModuleList([nn.Conv2d(input_chanel, output_chanel, kernel_size, padding="same") for i in range(num_nodes)])
        self.num_nodes = num_nodes
        self.binary_code = '0' if binary_code is None else binary_code
        while set(self.binary_code).issubset(set('0')):
            self.binary_code = ''.join([random.choice(['0', '1']) for _ in range(int(num_nodes * (num_nodes - 1) / 2))])
        self.inputs, self.outputs, self.separated_connections = self.get_nodes_connections(self.num_nodes, self.binary_code)

    @staticmethod
    def get_nodes_connections(nodes, connections):
        """
        This def get number of nodes and binary code string

        # Parameters
        1. `*nodes` : int
            - The number of nodes must be pass
        2. `*connections` : str
            - The connections between nodes 
            - The number of nodes must be pass

        # Returns
        - tuple
            - `inputs` : list
                - inputs connections between nodes
            - `outputs` : list
                - outputs connections between nodes
            - `separated_connections` : list
                - split binary code connections to encode of nodes
        """

        ctr = 0
        idx = 0

        separated_connections = []
        while idx + ctr < len(connections):
            ctr += 1
            separated_connections.append(connections[idx:idx + ctr])
            idx += ctr

        outputs = []
        for node in range(nodes - 1):
            node_outputs = []
            for i, node_connections in enumerate(separated_connections[node:]):
                if node_connections[node] == '1':
                    node_outputs.append(node + i + 1)
            outputs.append(node_outputs)
        outputs.append([])

        inputs = [[]]
        for node in range(1, nodes):
            node_inputs = []
            for i, connection in enumerate(separated_connections[node - 1]):
                if connection == '1':
                    node_inputs.append(i)
            inputs.append(node_inputs)
        return inputs, outputs, separated_connections

    def __build_stage(self, x):
        output_vars = []
        all_vars = [None] * self.num_nodes
        for i, (ins, outs) in enumerate(zip(self.inputs, self.outputs)):
            if ins or outs:
                if not ins:
                    tmp = x
                else:
                    add_vars = [all_vars[i] for i in ins]
                    if len(add_vars) > 1:
                        tmp = torch.sum(torch.stack(add_vars), dim=0)
                    else:
                        tmp = add_vars[0]
                tmp = F.relu(self.convs[i](tmp))
                all_vars[i] = tmp
                if not outs:
                    output_vars.append(tmp)
        if len(output_vars) > 1:
            return torch.sum(torch.stack(output_vars), dim=0)
        return output_vars[0]

    def forward(self, x):
        return self.__build_stage(x)
