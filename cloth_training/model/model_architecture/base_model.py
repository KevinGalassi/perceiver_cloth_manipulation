import torch
import torch.nn as nn
from cloth_training.model.model_architecture.model_utils import layer_init


class OneStageNetwork(nn.Module) :
   def __init__(self, input_dim, hidden_dim, out_dim1, out_dim2, device=None) -> None:
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


      self.input_dim = input_dim
      self.hidden_dim = hidden_dim
      self.out_dim1 = out_dim1
      self.out_dim2 = out_dim2


      self.l1 = nn.Linear(input_dim, 512).to(device)
      self.l2 = nn.Linear(512, 256).to(device)
      self.l3 = nn.Linear(256, hidden_dim).to(device)

      self.p1 = nn.Linear(hidden_dim, 256).to(device)
      self.p2 = nn.Linear(256, out_dim1).to(device)

      self.v1 = nn.Linear(hidden_dim, 64).to(device)
      self.v2 = nn.Linear(64, out_dim2).to(device)

      self.droupout = nn.Dropout()

      self.relu = nn.ReLU()


   def forward(self, x):
      x = self.l1(x)
      x = self.relu(x)
      x = self.droupout(x)
      x = self.l2(x)
      x = self.relu(x)     
      x = self.droupout(x)
      h = self.l3(x)

      p = self.p1(h)
      p = self.relu(p)  
      p = self.droupout(p)
      p = self.p2(p)

      a = self.v1(h)
      a = self.relu(a)
      a = self.droupout(a)
      a = self.v2(a)

      return p, a 




class InputEncoder(nn.Module):
   def __init__(self, input_dim, output_dim, device=None) -> None:
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


      self.input_dim = input_dim
      self.output_dim = output_dim


      self.l1 = layer_init(nn.Linear(input_dim, 512).to(device))
      self.l2 = layer_init(nn.Linear(512, 256).to(device))
      self.l3 = layer_init(nn.Linear(256, output_dim).to(device))
      self.droupout = nn.Dropout()


   def forward(self, x):
      x = self.l1(x).relu()
      x = self.droupout(x)
      x = self.l2(x).relu()
      x = self.droupout(x)
      x = self.l3(x)

      return x.relu()





class ProbDecoder(nn.Module) :
   def __init__(self, input_dim, output_dim, device=None) -> None:
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.l1 = layer_init(nn.Linear(input_dim, 256).to(device))
      self.l2 = layer_init(nn.Linear(256, output_dim).to(device))
      self.dropout = nn.Dropout()
  

   def forward(self, x):
      x = self.l1(x).relu()
      x = self.dropout(x)
      x = self.l2(x)
      return x
   



class OffsetNetwork3stage(nn.Module) :
   def __init__(self, input_dim, cat_dim, device=None) -> None:
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.l1 = layer_init(nn.Linear(input_dim + cat_dim, 256).to(device))
      self.l2 = layer_init(nn.Linear(256, 64).to(device))
      self.l3 = layer_init(nn.Linear(64, 2).to(device))
      self.dropout = nn.Dropout()
  

   def forward(self, x, cat):
      
      x = torch.cat([x, cat], dim=1)

      x = self.l1(x).relu()
      x = self.dropout(x)
      x = self.l2(x).relu()
      x = self.dropout(x)
      x = self.l3(x)
      return x
      return x.relu()
   


class OffsetNetworkECatFc(nn.Module) :
   def __init__(self, input_dim, cat_dim, device =None) -> None:
      '''
      input_dim: dimension of the input
      middle_dim: dimension of the layer after the initial encoder
      cat_dim: dimension of the tensor cat to the output of the encoder [n points]
      inj_hidden_dim: dimension of the hidden layer after the concatenation
      '''
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
      self.e = layer_init(nn.Linear(input_dim, 64).to(device))
      
      self.lin1 = layer_init(nn.Linear(64 + cat_dim, 256).to(device))
      self.lin2 = layer_init(nn.Linear(256, 64).to(device))
      self.lin3 = layer_init(nn.Linear(64, 2).to(device))

      self.dropout = nn.Dropout()



   def forward(self, x, cat) :

      h = self.e(x)

      z = torch.cat([h, cat], dim=1)
      z = self.lin1(z).relu()
      z = self.dropout(z)
      
      z = self.lin2(z).relu()
      z = self.dropout(z)

      z = self.lin3(z)
      
      return x

      return z.relu()
   

class OffsetNetworkDualCat(nn.Module) :
   def __init__(self, input_dim, middle_dim, cat_dim, inj_hidden_dim, device =None) -> None:
      '''
      input_dim: dimension of the input
      middle_dim: dimension of the layer after the initial encoder
      cat_dim: dimension of the tensor cat to the output of the encoder [n points]
      inj_hidden_dim: dimension of the hidden layer after the concatenation
      '''
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.e = layer_init(nn.Linear(input_dim, middle_dim).to(device))

      self.lin1 = layer_init(nn.Linear(cat_dim + middle_dim, inj_hidden_dim).to(device))
      self.lin2 = layer_init(nn.Linear(cat_dim + inj_hidden_dim, 32).to(device))
      self.lin3 = layer_init(nn.Linear(32, 2).to(device))
      
      self.dropout = nn.Dropout()

   def forward(self, x, cat) :

      h = self.e(x)
      x = torch.cat([h, cat], dim=1)
      x = self.lin1(x).relu()
      x = self.dropout(x)

      x = torch.cat([x, cat], dim=1)
      x = self.lin2(x).relu()
      x = self.dropout(x)

      x = self.lin3(x)


      return x

      return x.relu()
   


class OffsetNetworkDualEDualCat(nn.Module) :
   def __init__(self, input_dim, middle_dim, cat_dim, inj_hidden_dim, device =None) -> None:
      '''
      input_dim: dimension of the input
      middle_dim: dimension of the layer after the initial encoder
      cat_dim: dimension of the tensor cat to the output of the encoder [n points]
      inj_hidden_dim: dimension of the hidden layer after the concatenation
      '''
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.e1 = layer_init(nn.Linear(input_dim, 64).to(device))
      self.e2 = layer_init(nn.Linear(64, middle_dim).to(device))


      self.lin1 = layer_init(nn.Linear(cat_dim + middle_dim, inj_hidden_dim).to(device))
      self.lin2 = layer_init(nn.Linear(cat_dim + inj_hidden_dim, 32).to(device))
      self.lin3 = layer_init(nn.Linear(32, 2).to(device))

      self.dropout = nn.Dropout()

   def forward(self, x, cat) :

      h = self.e1(x).relu()
      h = self.dropout(h)
      h = self.e2(h).relu()
      h = self.dropout(h)

      x = torch.cat([h, cat], dim=1)
      x = self.lin1(x).relu()
      x = self.dropout(x)

      x = torch.cat([x, cat], dim=1)
      x = self.lin2(x).relu()
      x = self.dropout(x)
      
      x = self.lin3(x)
      return x
   


class OffsetNetworkObsEncodingDualCat(nn.Module) :
   def __init__(self, input_dim, middle_dim, cat_dim, inj_hidden_dim,obs_dim, device =None) -> None:
      '''
      input_dim: dimension of the input
      middle_dim: dimension of the layer after the initial encoder
      cat_dim: dimension of the tensor cat to the output of the encoder [n points]
      inj_hidden_dim: dimension of the hidden layer after the concatenation
      '''
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.e = layer_init(nn.Linear(input_dim, middle_dim).to(device))

      self.obs_encoder = layer_init(nn.Linear(obs_dim, 256).to(device))
      self.obs_encoder2 = layer_init(nn.Linear(256, 64).to(device))


      self.lin1 = layer_init(nn.Linear(cat_dim + middle_dim + 64, inj_hidden_dim).to(device))
      self.lin2 = layer_init(nn.Linear(cat_dim + inj_hidden_dim, 32).to(device))
      self.lin3 = layer_init(nn.Linear(32, 2).to(device))

      self.dropout = nn.Dropout()

   def forward(self, x, cat, obs) :

      h = self.e(x)

      e_o = self.obs_encoder(obs)
      e_o = self.dropout(e_o)
      e_o = self.obs_encoder2(e_o)
      e_o = self.dropout(e_o)

      x = torch.cat([h, cat, e_o], dim=1)
      x = self.lin1(x).relu()
      x = self.dropout(x)

      x = torch.cat([x, cat], dim=1)
      x = self.lin2(x).relu()
      x = self.dropout(x)

      x = self.lin3(x)

      return x
      return x.relu()
   



class OffsetNetworkObsEDualCat(nn.Module) :
   def __init__(self, input_dim, middle_dim, cat_dim, inj_hidden_dim, device =None) -> None:
      '''
      input_dim: dimension of the input
      middle_dim: dimension of the layer after the initial encoder
      cat_dim: dimension of the tensor cat to the output of the encoder [n points]
      inj_hidden_dim: dimension of the hidden layer after the concatenation
      '''
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.obs_e1 = layer_init(nn.Linear(input_dim, 512).to(device))
      self.obs_e2 = layer_init(nn.Linear(512, 256).to(device))
      self.obs_e3 = layer_init(nn.Linear(256, middle_dim).to(device))


      self.lin1 = layer_init(nn.Linear(cat_dim + middle_dim, inj_hidden_dim).to(device))
      self.lin2 = layer_init(nn.Linear(cat_dim + inj_hidden_dim, 32).to(device))
      self.lin3 = layer_init(nn.Linear(32, 2).to(device))
      
      self.dropout = nn.Dropout()

   def forward(self, x, cat) :


      e_o = self.obs_e1(x)
      e_o = self.dropout(e_o)
      e_o = self.obs_e2(e_o)
      e_o = self.dropout(e_o)
      h = self.obs_e3(e_o)

      print(h.shape)
      print(cat.shape)
      x = torch.cat([h, cat], dim=-1)
      x = self.lin1(x).relu()
      x = self.dropout(x)

      x = torch.cat([x, cat], dim=-1)
      x = self.lin2(x).relu()
      x = self.dropout(x)

      x = self.lin3(x)

      return x
      return x.relu()
   

class OffsetNetworkTriple(nn.Module) :
   def __init__(self, input_dim, middle_dim, cat_dim, device =None) -> None:
      '''
      input_dim: dimension of the input
      middle_dim: dimension of the layer after the initial encoder
      cat_dim: dimension of the tensor cat to the output of the encoder [n points]
      inj_hidden_dim: dimension of the hidden layer after the concatenation
      '''
      super().__init__()


      if device is None:
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.obs_e1 = layer_init(nn.Linear(input_dim, 512).to(device))
      self.obs_e2 = layer_init(nn.Linear(512, 256).to(device))


      self.lin1 = layer_init(nn.Linear(256 + middle_dim + cat_dim, 256).to(device))
      self.lin2 = layer_init(nn.Linear(256, 32).to(device))
      self.lin3 = layer_init(nn.Linear(32, 2).to(device))
      
      self.dropout = nn.Dropout()

   def forward(self, x,h,p) :


      e_o = self.obs_e1(x)
      e_o = self.dropout(e_o)
      e_o = self.obs_e2(e_o)
      e_o = self.dropout(e_o)

      z = torch.cat([e_o, h, p], dim=1)

      x = self.lin1(x).relu()
      x = self.dropout(x)
      x = self.lin2(x).relu()
      x = self.dropout(x)

      x = self.lin3(x)

      return x




