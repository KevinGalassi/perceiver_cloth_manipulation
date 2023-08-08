import torch

import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

from cloth_training.model.model_architecture.model_utils import layer_init
import numpy as np
import torch.nn.functional as F

from einops import rearrange, repeat

def set_seed(seed):
   if seed == None :
      print('Warning : seed is None')
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True




class Attention(nn.Module):
   def __init__(self, embed_dim, num_heads, batch_first=True, dropout=0.1) :
      super(Attention, self).__init__()
      self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=batch_first)
      
      self.ff1=nn.Linear(embed_dim, embed_dim*2)
      self.activation=nn.GELU()
      self.ff2=nn.Linear(embed_dim*2, embed_dim)
      self.norm1=nn.LayerNorm(embed_dim)
      self.norm2=nn.LayerNorm(embed_dim)
      self.dropout=nn.Dropout(dropout)

   def forward(self, q, context = None):
      context = default(context, q)
      
      mha, att = self.multihead_attention(q, context, context)
      q = self.norm1(q + self.dropout(mha))
      lin= self.dropout(self.ff2(self.activation(self.ff1(q))))
      q = self.norm2(q + lin)
      return q

def exists(val):
   return val is not None

def default(val, d):
   return val if val is not None else d

from torch import einsum
class Attention2(nn.Module):

   def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
      super().__init__()
      inner_dim = dim_head * heads
      context_dim = default(context_dim, query_dim)

      self.scale = dim_head ** -0.5
      self.heads = heads

      self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
      self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

      self.dropout = nn.Dropout(dropout)
      self.to_out = nn.Linear(inner_dim, query_dim)

   def forward(self, x, context = None):
      h = self.heads

      q = self.to_q(x)
      context = default(context, x)
      k, v = self.to_kv(context).chunk(2, dim = -1)

      q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

      sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

      # attention, what we cannot get enough of
      attn = sim.softmax(dim = -1)
      attn = self.dropout(attn)

      out = einsum('b i j, b j d -> b i d', attn, v)
      out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
      return self.to_out(out)



class GEGLU(nn.Module):
   

   def forward(self, x):
      x, gates = x.chunk(2, dim = -1)
      return x * F.gelu(gates)


class FeedForward(nn.Module):  
   def __init__(self, dim, mult = 4, dropout = 0.):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(dim, dim * mult * 2),
         GEGLU(),
         nn.Linear(dim * mult, dim),
         nn.Dropout(dropout)
      )

   def forward(self, x):
      return self.net(x)




class TransformerNetwork(nn.Module) : 
   def __init__(self, hidden_dim, num_heads, num_layers, dropout=0.1, seed=None ) -> None :
      super().__init__()

      if seed is not None:
         set_seed(seed)

      self.input_embedding = nn.Linear(3, hidden_dim)

      # Encoder transformer layer
      encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout, batch_first=True)
      encoder_norm = LayerNorm(hidden_dim)
      self.transformer = TransformerEncoder(encoder_layer,num_layers=num_layers, norm=encoder_norm)

     
      self.o1 = layer_init(nn.Linear(hidden_dim, 64))
      self.o2 = layer_init(nn.Linear(64, 1))

      self.dropout = nn.Dropout(dropout)



   def forward(self, x):
      # Features embedding
      e = self.input_embedding(x)
      
      # Transformer
      z = self.transformer(e)

      # Output
      #o = nn.functional.elu(self.o1(z))
      #o = self.dropout(o)
      #p = self.o2(o)

      o = self.o1(z)
      p = self.o2(o)

      return p



import torch
import torch.nn as nn

class Perceiver(nn.Module):
   def __init__(self, depth = 3,
                     input_dim = 3, 
                     input_embedding_dim = 200,
                     num_latents = 200,
                     num_cross_heads = 1,
                     num_output_heads = 1, 
                     num_latent_heads = 6,
                     num_latent_layers = 6,
                     seed = None ):
      
      super(Perceiver, self).__init__()
      """

      The shape of the final attention mechanism will be:
        depth * (cross attention -> latent_self_attention*num_latent_layers)


      input_dim: dimension of initial input (C)
      latent_dim: dimension of latent space of the model (D)
      num_latents: reduced number of latents (N)

      """
      
      self.num_latent_layers = num_latent_layers

      self.depth = depth

      set_seed(seed) if seed is not None else None


      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(input_dim, input_embedding_dim),
         nn.ReLU(),
      )

      # Latent Array
      self.latents = nn.Parameter(torch.normal(0, 0.2, (num_latents, input_embedding_dim)))

   
      # Latent trasnformer module
      self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      self.self_latent_ff = FeedForward(input_embedding_dim)

      # Cross-attention module
      self.cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_cross_heads, batch_first=True)
      self.cross_ff = FeedForward(input_embedding_dim)
      
      # Output cross-attention of latent space with input as query      
      self.output_cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_output_heads, batch_first=True)
      
      # Decoder
      self.output_layer = nn.Sequential(
               nn.Linear(input_embedding_dim, input_embedding_dim//2),
               nn.ReLU(),
               nn.Linear(input_embedding_dim//2, 1)
            )
 

   def forward(self, inputs):

      data = self.input_embedding(inputs)

      b = inputs.shape[0]
      x = repeat(self.latents, 'n d -> b n d', b = b)

      for _ in range(self.depth):
         x = self.cross_attention(x, context = data) + x
         x = self.cross_ff(x) + x

         for _ in range(self.num_latent_layers):
            x = self.self_latent_attention(x) + x
            x = self.self_latent_ff(x) + x

      # Output cross attention
      p = self.output_cross_attention(data, context = x)

      p = self.output_layer(p)

      return p
    



class PerceiverIO(nn.Module) :

   def __init__(self, depth,
                     input_dim, 
                     input_embedding_dim,
                     output_dim, 
                     num_latents, 
                     num_cross_heads, 
                     num_output_heads, 
                     num_latent_heads,
                     num_latent_layers = 6,
                     seed = None,) :
      """
      depth : number of cross-attention + latent self-attention layers
      input_dim : dimension of initial input (C)
      input_embedding_dim : dimension of input embedding (D)
      output_dim : dimension of query output (O)
      num_latents : reduced number of latents (N)


      """
      super(PerceiverIO, self).__init__()

      set_seed(seed) if seed is not None else None

      self.perceiver_layer = Perceiver(depth = depth,
                                       input_dim = input_dim, 
                                       input_embedding_dim = input_embedding_dim,
                                       output_dim = output_dim, 
                                       num_latents = num_latents,
                                       num_cross_heads = num_cross_heads, 
                                       num_latent_heads = num_latent_heads,
                                       num_latent_layers = num_latent_layers,
                                       seed = seed)
                                 
      # Cross-attention module
      self.output_query = nn.Parameter(torch.normal(0, 0.2, (output_dim, input_embedding_dim)))
      self.output_cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_output_heads, batch_first=True)
      

      self.output_ff = nn.Linear(input_embedding_dim, 1)



   def forward(self, inputs):


      x = self.perceiver_layer(inputs, ff_layer=False)

      b = inputs.shape[0]
      o = repeat(self.output_query, 'n d -> b n d', b = b)
      p = self.output_cross_attention(o, x, x)

      p = self.output_ff(p)

      return p
   




class PerceiverIOv2(nn.Module) :

   def __init__(self, depth,
                     input_dim, 
                     input_embedding_dim,
                     output_dim, 
                     num_latents, 
                     num_cross_heads, 
                     num_output_heads, 
                     num_latent_heads,
                     num_latent_layers = 6,
                     seed = None,) :
      """
      depth : number of cross-attention + latent self-attention layers
      input_dim : dimension of initial input (C)
      input_embedding_dim : dimension of input embedding (D)
      output_dim : dimension of query output (O)
      num_latents : reduced number of latents (N)


      """
      super(PerceiverIOv2, self).__init__()

      set_seed(seed) if seed is not None else None

      self.perceiver_layer = Perceiver(depth = depth,
                                       input_dim = input_dim, 
                                       input_embedding_dim = input_embedding_dim,
                                       output_dim = output_dim, 
                                       num_latents = num_latents,
                                       num_cross_heads = num_cross_heads, 
                                       num_latent_heads = num_latent_heads,
                                       num_latent_layers = num_latent_layers,
                                       seed = seed)


      self.num_latent_layers = num_latent_layers
      self.depth = depth

      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(input_dim, input_embedding_dim),
         nn.ReLU(),
      )          

      # Latent Array
      #self.latents = nn.Parameter(torch.randn(num_latents, input_embedding_dim))
      

      self.latents = nn.Parameter(torch.normal(0, 0.2, (num_latents, input_embedding_dim)))
      
      # Cross-attention module

      self.cross_attention = Attention2(input_embedding_dim, heads=num_cross_heads)
      self.cross_ff = FeedForward(input_embedding_dim)
      
      self.self_latent_attention = Attention2(input_embedding_dim, heads=num_latent_heads)
      self.self_latent_ff = FeedForward(input_embedding_dim)

      # Cross-attention module
      self.output_cross_attention = Attention2(input_embedding_dim, heads=num_output_heads)
      self.output_ff = nn.Linear(input_embedding_dim, 1)


   

   def forward(self, inputs):

      data = self.input_embedding(inputs)


      b = inputs.shape[0]
      x = repeat(self.latents, 'n d -> b n d', b = b)

      # Input cross attention
      for _ in range(self.depth):
         
         x = self.cross_attention(x, context = data) + x
         x = self.cross_ff(x) + x

         for _ in range(self.num_latent_layers):
            x = self.self_latent_attention(x,)
            x = self.self_latent_ff(x) + x

      p = self.output_cross_attention(data, context = data) + data
      p = self.output_ff(p)

      return p
   

   def input_cross_attention(self, inputs) :

      data = self.input_embedding(inputs)
      b = inputs.shape[0]
      x = repeat(self.latents, 'n d -> b n d', b = b)
      x = self.cross_attention(x, data, data)
   
      return x

class OffsetPerceiver(nn.Module) :

   def __init__(self, depth,
                     input_query_dim,    # Dimension of Prob.Features
                     input_embedding_dim,
                     output_dim, 
                     num_cross_heads, 
                     num_output_heads, 
                     num_latent_heads,
                     num_latent_layers,
                     seed = None,) :
      super(OffsetPerceiver, self).__init__()

      self.num_latent_layers = num_latent_layers
      self.depth = depth
      set_seed(seed) if seed is not None else None


      # Input Feature Embedding
      self.input_embedding = nn.Sequential(
         nn.Linear(input_query_dim, int(input_embedding_dim // 2)),
         nn.ReLU(),
         nn.Linear(int(input_embedding_dim // 2), input_embedding_dim),
         nn.ReLU(),
      )

      # Latent trasnformer module
      self.self_latent_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_latent_heads, batch_first=True)
      
      # Cross-attention module
      self.cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_cross_heads, batch_first=True)
      
      # Output Cross Attention

      self.output_query = nn.Parameter(torch.normal(0, 0.2, (output_dim, input_embedding_dim)))
      self.output_cross_attention = Attention(embed_dim=input_embedding_dim, num_heads=num_output_heads, batch_first=True)
      

   


      self.output_ff = nn.Sequential(
         nn.Linear(input_embedding_dim, int(input_embedding_dim // 2)),
         nn.ReLU(),
         nn.Linear(int(input_embedding_dim // 2), 1),
         nn.ReLU(),
      )




   def forward(self, prob, q):
      '''
      prob = probability vector, output of the previous layer
      q = query vector, output of the first cross attention 

      '''

      # Input embedding of the probability vector to match dimension
      k = self.input_embedding(prob)
      b = q.shape[0]

      # Cross attention with input query
      for _ in range(self.depth):
         
         x = self.cross_attention(q, k, k)

         for _ in range(self.num_latent_layers):
            x = self.self_latent_attention(x, x, x)

      
      # Output cross attention
      o = repeat(self.output_query, 'n d -> b n d', b = b)
      p = self.output_cross_attention(o, x, x)

      # Output FF to bring dimension to 1
      p = self.output_ff(p)

      return p