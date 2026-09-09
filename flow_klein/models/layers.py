import torch

class node_mlp(torch.nn.Module):
    """
    This layer apply a chain of mlp on each node of tthe graph.
    thr input is a matric matrrix with n rows whixh n is the nide number.
    """
    def __init__(self, input, layers= [16, 16], normalize = False, dropout_rate = 0):
        """

        :param input: the feture size of input matrix; Number of the columns
        :param normalize: either use the normalizer layer or not
        :param layers: a list which shows the ouyput feature size of each layer; Note the number of layer is len(layers)
        """
        super(node_mlp, self).__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(input, layers[0])])

        for i in range(len(layers)-1):
            self.layers.append(torch.nn.Linear(layers[i],layers[i+1]))

        self.norm_layers = None
        if normalize:
            self.norm_layers =  torch.nn.ModuleList([torch.nn.BatchNorm1d(c) for c in [input]+layers])
        self.dropout = torch.nn.Dropout(dropout_rate)
        # self.reset_parameters()

    def forward(self, in_tensor, activation = torch.tanh, applyActOnTheLastLyr=True):
        h = in_tensor
        for i in range(len(self.layers)):
            if self.norm_layers!=None:
                if len(h.shape)==2:
                    h = self.norm_layers[i](h)
                else:
                    shape = h.shape
                    h= h.reshape(-1, h.shape[-1])
                    h = self.norm_layers[i](h)
                    h=h.reshape(shape)
            h = self.dropout(h)
            h = self.layers[i](h)
            if i != (len(self.layers)-1) or applyActOnTheLastLyr:
                h = activation(h)
        return h
