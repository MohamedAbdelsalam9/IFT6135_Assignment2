import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
    '''
    A helper function for producing N identical layers (each with their own parameters).

    inputs:
        module: a pytorch nn.module
        N (int): the number of copies of that module to return

    returns:
        a ModuleList with the copies of the module (the ModuleList is itself also a module)
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
is_cuda = torch.cuda.is_available()

# used to create a recurrent layer
# taking as input the cell object which contains the computation of 1 cell
# can be gru cell or rnn cell
# includes dropout on the input
class RNNLayer(nn.Module):
    def __init__(self,recurrent_cell, seq_len, dp_keep_prob):
        super(RNNLayer, self).__init__()
        self.seq_len = seq_len
        self.recurrent_cells = recurrent_cell#clones(recurrent_cell , seq_len)
        self.dropout = nn.Dropout(1-dp_keep_prob)
        self.init_weights()

    def init_weights(self):
        self.recurrent_cells.init_weights()

    def forward(self,input, hidden_state):
        out_hidden = [hidden_state]
        input = self.dropout(input)
        for word in range(self.seq_len):
            hidden_state_out = self.recurrent_cells.forward( input[word,:,:], out_hidden[-1])
            out_hidden.append(hidden_state_out)
        out_hidden.pop(0)
        hidden_state = torch.stack(out_hidden)
        return  hidden_state

# includes recurrent cell computation for a single word processing
# and returns its hidden state to be sent to the next recurrent cell
class RnnCell(nn.Module):
    def __init__(self, hidden_size, embed_size ):
        super(RnnCell, self).__init__()
        self.hidden_transform = nn.Linear(embed_size + hidden_size , hidden_size)

        self.activation = nn.Tanh()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.init_weights()


    def init_weights(self):
        init_bound = math.sqrt(1.0/self.hidden_size)
        self.hidden_transform.weight.data.uniform_(-init_bound,init_bound)
        self.hidden_transform.bias.data.uniform_(-init_bound,init_bound)

    def forward(self, input, hidden_state):
        #input = self.input_transform(input)
        #hidden_state = input + self.hidden_transform(hidden_state)
        hidden_state = self.hidden_transform(torch.cat((input, hidden_state), 1) )
        hidden_state = self.activation(hidden_state)
        return  hidden_state


class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    """
    emb_size:     The number of units in the input embeddings
    hidden_size:  The number of hidden units per layer
    seq_len:      The length of the input sequences
    vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                  each time-step)
    dp_keep_prob: The probability of *not* dropping out units in the 
                  non-recurrent connections.
                  Do not apply dropout on recurrent connections.
    """
    super(RNN, self).__init__()

    # TODO ========================
    # Initialization of the parameters of the recurrent and fc layers. 
    # Your implementation should support any number of stacked hidden layers 
    # (specified by num_layers), use an input embedding layer, and include fully
    # connected layers with dropout after each recurrent layer.
    # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
    # modules, but not recurrent modules.
    #
    # To create a variable number of parameter tensors and/or nn.Modules 
    # (for the stacked hidden layer), you may need to use nn.ModuleList or the 
    # provided clones function (as opposed to a regular python list), in order 
    # for Pytorch to recognize these parameters as belonging to this nn.Module 
    # and compute their gradients automatically. You're not obligated to use the
    # provided clones function.
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(vocab_size, emb_size)
    rnn_cell = RnnCell(hidden_size, emb_size)
    first_rnn_layer = RNNLayer(rnn_cell, seq_len,dp_keep_prob)
    rnn_cell = RnnCell(hidden_size, hidden_size)
    rnn_layer = RNNLayer(rnn_cell, seq_len,dp_keep_prob)
    self.rnn = clones(rnn_layer,num_layers-1)
    self.rnn.insert(0,copy.deepcopy(first_rnn_layer))
    self.output = torch.nn.Linear(hidden_size, vocab_size)
    self.dropout = nn.Dropout(1-dp_keep_prob)
    self.init_weights()

  def init_weights(self):
    # TODO ========================
    # Initialize the embedding and output weights uniformly in the range [-0.1, 0.1]
    # and the embedding and output biases to 0 (in place).
    # Initialize all other (i.e. recurrent and linear) weights AND biases uniformly
    # in the range [-k, k] where k is the square root of 1/hidden_size
    nn.init.uniform_(self.embedding.weight.data,a=-0.1,b=0.1)
    self.output.weight.data.uniform_(-0.1,0.1)
    nn.init.zeros_(self.output.bias.data)
    for layer_indx in range(self.num_layers):
        self.rnn[layer_indx].init_weights()

  def init_hidden(self):
    # TODO ========================
    # initialize the hidden states to zero
    """
    This is used for the first mini-batch in an epoch, only.
    """
    h = torch.nn.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
    if is_cuda:
        h = h.cuda()
    return h

  def forward(self, inputs, hidden):
    # TODO ========================
    # Compute the forward pass, using nested python for loops.
    # The outer for loop should iterate over timesteps, and the 
    # inner for loop should iterate over hidden layers of the stack. 
    # 
    # Within these for loops, use the parameter tensors and/or nn.modules you 
    # created in __init__ to compute the recurrent updates according to the 
    # equations provided in the .tex of the assignment.
    #
    # Note that those equations are for a single hidden-layer RNN, not a stacked
    # RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
    # inputs to to the {l+1}-st layer (taking the place of the input sequence).

    """
    Arguments:
        - inputs: A mini-batch of input sequences, composed of integers that 
                    represent the index of the current token(s) in the vocabulary.
                        shape: (seq_len, batch_size)
        - hidden: The initial hidden states for every layer of the stacked RNN.
                        shape: (num_layers, batch_size, hidden_size)
    
    Returns:
        - Logits for the softmax over output tokens at every time-step.
              **Do NOT apply softmax to the outputs!**
              Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
              this computation implicitly.
                    shape: (seq_len, batch_size, vocab_size)
        - The final hidden states for every layer of the stacked RNN.
              These will be used as the initial hidden states for all the 
              mini-batches in an epoch, except for the first, where the return 
              value of self.init_hidden will be used.
              See the repackage_hiddens function in ptb-lm.py for more details, 
              if you are curious.
                    shape: (num_layers, batch_size, hidden_size)
    """
    embed_output = self.embedding(inputs) # nwords, batchsize , embed size
    # convert its size for faster RNN
    input_tmp = embed_output
    hidden_data = []
    for layer in range(self.num_layers):
        input_tmp = self.rnn[layer](input_tmp, hidden[layer])
        hidden_data.append(input_tmp[-1,:,:])
    hidden = torch.stack(hidden_data)
    input_tmp = self.dropout(input_tmp)
    logits = self.output(input_tmp)
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    temperature = 10
    samples = np.zeros((generated_seq_len, input.size()[1])) 
    batch_size = input.size()[1]
    logits = input
    for seq_indx in range(0,  generated_seq_len):
        logits, hidden = self.forward(input, hidden) 
        logits =  logits.squeeze().div(temperature).exp().cpu()
        # it makes the output richer. 
        # Sticking to the most probable words would restrict 
        # the model to always use the most commonly used words
        word_indexes = torch.multinomial(logits, batch_size).numpy()
        samples[seq_indx,:] = word_indexes
        input = torch.unsqueeze(torch.from_numpy(word_indexes),0)
        if is_cuda:
            input = input.cuda()
    return samples


# Problem 2

# class used to make a single gru celll that returns its hidden state sent to the next cell
# in the RNNLayer Object
class GruCell(nn.Module):
    def __init__(self, hidden_size, embed_size ):
        super(GruCell, self).__init__()
        self.update_gate = nn.Linear(embed_size + hidden_size, hidden_size)
        self.activation_update_reset = nn.Sigmoid()
        self.activation_hidden_memory = nn.Tanh()
        self.reset_gate = nn.Linear(embed_size + hidden_size , hidden_size)

        self.hidden_transform = nn.Linear(embed_size + hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.init_weights()


    def init_weights(self):
        init_bound = np.sqrt(1.0/self.hidden_size)

        self.update_gate.weight.data.uniform_(-init_bound,init_bound)
        self.update_gate.bias.data.uniform_(-init_bound,init_bound)

        self.reset_gate.weight.data.uniform_(-init_bound,init_bound)
        self.reset_gate.bias.data.uniform_(-init_bound,init_bound)

        self.hidden_transform.weight.data.uniform_(-init_bound,init_bound)
        self.hidden_transform.bias.data.uniform_(-init_bound,init_bound)

    def forward(self, input, hidden_state):
        update_gate_output = self.update_gate(torch.cat((input, hidden_state),1))
        update_gate_output = self.activation_update_reset(update_gate_output)

        reset_gate_output = self.reset_gate(torch.cat((input,hidden_state),1))
        reset_gate_output = self.activation_update_reset(reset_gate_output)

        current_hidden = self.hidden_transform(torch.cat((input,torch.mul(reset_gate_output, hidden_state)),1))
        current_hidden = self.activation_hidden_memory(current_hidden)

        final_hidden = torch.mul((1-update_gate_output),hidden_state) + torch.mul(update_gate_output,current_hidden)

        return  final_hidden


class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for 
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size

    self.embedding = nn.Embedding(vocab_size, emb_size)
    gru_cell = GruCell(hidden_size, emb_size)
    first_gru_layer = RNNLayer(gru_cell, seq_len,dp_keep_prob)
    gru_cell = GruCell(hidden_size, hidden_size)
    gru_layer = RNNLayer(gru_cell, seq_len,dp_keep_prob)
    self.gru = clones(gru_layer,num_layers-1)
    self.gru.insert(0,copy.deepcopy(first_gru_layer))
    self.output = torch.nn.Linear(hidden_size, vocab_size)
    self.dropout = nn.Dropout(1-dp_keep_prob)
    self.init_weights_uniform()

  def init_weights_uniform(self):
    nn.init.uniform_(self.embedding.weight.data,a=-0.1,b=0.1)
    self.output.weight.data.uniform_(-0.1,0.1)
    nn.init.zeros_(self.output.bias.data)
    for layer_indx in range(self.num_layers):
        self.gru[layer_indx].init_weights()

  def init_hidden(self):
    # TODO ========================
    h = torch.nn.Parameter(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
    if is_cuda:
        h = h.cuda()
    return h

  def forward(self, inputs, hidden):
    # TODO ========================
    embed_output = self.embedding(inputs) # nwords, batchsize , embed size
    input_tmp = embed_output
    hidden_data = []
    for layer in range(self.num_layers):
        input_tmp = self.gru[layer](input_tmp, hidden[layer])
        hidden_data.append(input_tmp[-1,:,:])
    hidden = torch.stack(hidden_data)
    input_tmp = self.dropout(input_tmp)
    logits = self.output(input_tmp)
    return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
    temperature = 10
    samples = np.zeros((generated_seq_len, input.size()[1])) 
    batch_size = input.size()[1]
    logits = input
    for seq_indx in range(0,  generated_seq_len):
        logits, hidden = self.forward(input, hidden) 
        logits =  logits.squeeze().div(temperature).exp().cpu()
        word_indexes = torch.multinomial(logits, batch_size).numpy()
        samples[seq_indx,:] = word_indexes
        input = torch.unsqueeze(torch.from_numpy(word_indexes),0)
        if is_cuda:
            input = input.cuda()
    return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of input and output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units 
        self.n_heads = n_heads

        # TODO: create/initialize any necessary parameters or layers
        # Initialize all weights and biases uniformly in the range [-k, k],
        # where k is the square root of 1/n_units.
        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout
        # ETA: you can also use softmax
        # ETA: you can use the "clones" function we provide.
        # ETA: you can use masked_fill

        self.linear_q = nn.Linear(self.n_units, self.n_units, bias=True)
        self.linear_k = nn.Linear(self.n_units, self.n_units, bias=True)
        self.linear_v = nn.Linear(self.n_units, self.n_units, bias=True)
        self.linear_o = nn.Linear(self.n_units, self.n_units, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()


    def init_weights(self):
        k = 1 / math.sqrt(self.n_units)
        nn.init.uniform_(self.linear_q.weight, -k, k)
        nn.init.uniform_(self.linear_k.weight, -k, k)
        nn.init.uniform_(self.linear_v.weight, -k, k)
        nn.init.uniform_(self.linear_o.weight, -k, k)
        nn.init.zeros_(self.linear_q.bias.data)
        nn.init.zeros_(self.linear_k.bias.data)
        nn.init.zeros_(self.linear_v.bias.data)
        nn.init.zeros_(self.linear_o.bias.data)

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value correspond to Q, K, and V in the latex, and
        # they all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. a_i in the .tex)
        # Also apply dropout to the attention values.
        #mask = mask.to(dtype=torch.float32)
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        query_i = self.linear_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).permute(2, 0, 1, 3)
        key_i = self.linear_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).permute(2, 0, 1, 3)
        value_i = self.linear_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).permute(2, 0, 1, 3)

        a_i = torch.matmul(query_i, key_i.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = mask.unsqueeze(0)
        a_i = a_i.masked_fill_(~mask, 0)
        #a_i = a_i * mask - 10**9 * (1 - mask)
        a_i = torch.exp(a_i - a_i.max(dim=-1)[0].unsqueeze(-1))
        a_i = a_i / torch.sum(a_i, -1, keepdim=True)
        heads = torch.matmul(a_i, value_i)
        heads = heads.permute(1, 2, 0, 3).reshape((batch_size, seq_len, -1))
        a = self.linear_o(heads)
        a = self.dropout(a)
        return a    # size: (batch_size, seq_len, self.n_units)






#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)
        
    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6, 
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)
    
    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

