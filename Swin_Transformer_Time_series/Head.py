import torch
import torch.nn as nn

import torch
import torch.nn as nn
#class PredictionHead(nn.Module):
#    def __init__(self, dim, c_in, head_dropout=0.4):
#        super().__init__()
#        self.dropout = nn.Dropout(head_dropout)
#        self.layerNorm = nn.LayerNorm(dim)
#        self.drop = nn.Dropout(0.4)
#        self.linear2a = nn.Linear(dim*16, 96)  # Assuming you want 96 time steps in the output
#        # Change the output features of linear2b to 96 instead of 7
#        self.linear2b = nn.Linear(96, 96)  # Maintain the 96 time steps in the output

#    def forward(self, x):
#        x = self.layerNorm(x)
#        x = self.drop(x)
#        x = x.view(x.size(0), -1)  # Flatten the tensor keeping the batch dimension
#        x = self.linear2a(x)
#        x = self.linear2b(x)
#        # Now reshape to (batch_size, sequence_length, 1)
#        x = x.view(x.size(0), 96, 1)

#        return x

#class PredictionHead(nn.Module):
#    def __init__(self, dim, feature_index, head_dropout=0.4):
#        super().__init__()
#        self.feature_index = feature_index  # Index of the feature to predict
#        self.dropout = nn.Dropout(head_dropout)
#        self.linear = nn.Linear(dim, 1)  # Output one feature
#        self.layerNorm = nn.LayerNorm(dim)
#        self.activation = nn.GELU()  # Choose your preferred activation function
#        self.drop = nn.Dropout(0.4)
#        self.linear2a = nn.Linear(dim*16, 60)
#        self.linear2b = nn.Linear(60, 1)  # Output one feature

#    def forward(self, x):
#        print(x.shape)
#        x = self.layerNorm(x)
#        #x = self.activation(x)
#        print('layerNorm',x.shape)
#        x = self.drop(x)
#        x = x.view(x.size(0), -1)  # Flatten the tensor keeping the batch dimension
#        print('view',x.shape)

#        x = self.linear2a(x)
#        print('linear2a',x.shape)
#        x = self.linear2b(x)
#        print('linear2b',x.shape)
#        x = x.view(x.size(0), 96, 1)  # Reshape to (batch_size, sequence_length, 1)
#        print('view',x.shape)

#        return x


#class PredictionHead(nn.Module):
#    def __init__(self, classes, dim, head_dropout=0.3):
#        super().__init__()
#        self.classes = classes
        
#        # Regularization
#        self.batch_norm = nn.BatchNorm1d(64)
#        self.dropout1 = nn.Dropout(head_dropout)
#        self.dropout2 = nn.Dropout(head_dropout)
        
#        # Network Depth and Width
#        self.linear1 = nn.Linear(dim, dim*2)
#        self.linear2 = nn.Linear(dim*2, dim)
        
#        # For skip connection
#        self.linear_skip = nn.Linear(dim, dim)
        
#        # Final layers
#        self.linear3 = nn.Linear(dim*64, 100)
#        self.linear4 = nn.Linear(100, classes)
        
#        # Activation Functions
#        self.gelu = nn.GELU()
#        self.leaky_relu = nn.LeakyReLU()
        
#        # For dynamic reshaping
#        self.bs = None

#    def forward(self, x):
#        self.bs = x.size(0)
#        #print(x.shape)#torch.Size([32, 36, 256])
        
#        ## Batch Normalization
#        x = self.batch_norm(x)
    
    
#        ## Dynamically adjust the BatchNorm1d layer based on input size
#        #if hasattr(self, 'batch_norm'):
#        #    if self.batch_norm.num_features != x.size(1):
#        #        self.batch_norm = nn.BatchNorm1d(x.size(1)).to(x.device)
#        #else:
#        #    self.batch_norm = nn.BatchNorm1d(x.size(1)).to(x.device)
    

#        #print(x.shape)
#        # Initial Processing
#        #x = self.gelu(x)
#        x = self.dropout1(x)
#        #print(x.shape)

#        # Skip Connection
#        skip = self.linear_skip(x)
#        #print('skip',skip.shape)
#        # Network Depth and Width with Skip Connection
#        #x = self.leaky_relu(self.linear1(x))
#        #print('leaky_relu',x.shape)
#        x = self.dropout2(x)
#        #print(x.shape)
#        #x = self.leaky_relu(self.linear2(x) + skip)  # Add skip connection
#        #print('linear2',x.shape)
        
#        # Flatten feature dimensions
#        x = x.view(self.bs, -1) 
#        print('view1',x.shape)
#        # Final Dense Layers
#        #x = self.leaky_relu(self.linear3(x))
#        x = self.linear4(x)
#        #print('linear4',x.shape)
#        # Reshape to desired output shape
#        x= x.view(self.bs, 7, 96)
#        #x = x.view(self.bs, -1, self.classes)
#        #print('view1',x.shape)
#        x = x.transpose(1, 2)
#        return x




#class PredictionHead(nn.Module):
#    def __init__(self, classes, dim, head_dropout=0):
#        super().__init__()
#        self.classes = classes
#        self.dropout = nn.Dropout(head_dropout)
#        self.layerNorm = nn.LayerNorm(dim)
#        self.gelu = nn.GELU()
#        self.drop = nn.Dropout(0.4)
        
#        reduced_dim = dim // 2  # Reduce the dimensionality
#        self.linear = nn.Linear(reduced_dim * 64, classes)

#    def forward(self, x):
#        x = self.layerNorm(x)
#        x = self.gelu(x)
#        x = self.drop(x)
#        x = x.view(64, 64*256)  # Flatten all dimensions except the batch size
#        x = self.linear(x)
#        x = x.view(64, 720, 7)
#        return x


import torch
import torch.nn.functional as F

class CustomActivation(torch.nn.Module):
    def __init__(self, floor_value=-4.0):
        super().__init__()
        self.floor_value = floor_value

    def forward(self, x):
        # Apply a ReLU on the negative side to create a floor effect at -4
        return F.relu(x - self.floor_value) + self.floor_value




class PredictionHead(nn.Module):
    def __init__(self, classes, dim, head_dropout=0.4):
        super().__init__()
        self.custom_activation = CustomActivation(floor_value=-10.0)

        self.classes = classes
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(dim, classes) # you should adjust the input features to match the input matrix
        self.layerNorm = nn.LayerNorm(dim)
        #self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.prelu = nn.PReLU()
        self.drop = nn.Dropout(0.2) # was 0.3
        self.drop2=nn.Dropout(0.3)
        self.linear2a = nn.Linear(dim*64, 30)
        self.linear2b = nn.Linear(30, classes)
        #self.linear2 = nn.Linear(dim*64, classes) # you should adjust the input features to match the input matrix




    def forward(self, x):
        #print (x.shape)
        #    nn.LayerNorm(dim)
        #    nn.GELU(),
        #    GlobalAvgPool(),
        #    nn.Dropout(p_drop),
        #    nn.Linear(dim, classes)
        
        x = self.layerNorm(x)
        #x = self.custom_activation(x)
        #x = self.gelu(x)
        #x = self.leaky_relu(x)
        #x=self.prelu (x)
        #self.relu = nn.ReLU()
        #x = x.mean(dim=-2)
        x = self.drop(x)
        #x=self.drop2(x)
        #print (x.shape)       
        x = x.view(64 , 64*256 ) #
        #x = self.linear2(x)
        x = self.linear2a(x)
        x = self.linear2b(x)
       
        #print (x.shape)
        x = x.view(64,720,7)
        return x



    #def forward_musleh(self, x):
    #    #print("Input shape:", x.shape) # torch.Size([32, 64, 256])

    #    x = self.dropout(x) # torch.Size([32, 64, 256])
    #    #print("Shape after dropout:", x.shape)

    #    x = self.linear(x) # Shape after linear: ([32, 64, 672])
    #    #print("Shape after linear:", x.shape)

    #    x = x.mean(dim=-2) #Shape after mean: torch.Size([32, 672])
    #    #print("Shape after mean:", x.shape)

    #    x = x.unsqueeze(1) # Final Output shape: torch.Size([32, 1, 672])
    #    #print("Final Output shape:", x.shape)

    #    # Convert output to [32, 96, 7]

    #    #x = x.view(64, 7, 96)
    #    x = x.view(32, 720, 7)
    #    #print(" view shape:", x.shape)

    #    #x = x.transpose(1, 2)
    #    #print(" transpose shape:", x.shape)

    #    return x

    
#class PredictionHead(nn.Module):
#    def __init__(self,classes,dim, num_patches, n_vars=7, head_dropout=0, flatten=False):#n_vars, d_model, num_patch, forecast_len,
#        super().__init__()
#        self.flatten = flatten
#        head_dim = dim*16
#        self.gelu = nn.GELU()

#        self.flatten = nn.Flatten(start_dim=-2)
#        self.linear = nn.Linear( head_dim, classes )
#        self.dropout = nn.Dropout(head_dropout)


#    def forward(self, x): 
#        #print(x.shape)#torch.Size([64, 7, 128, 31])  , torch.Size([32, 36, 256])
#        """
#        x: [bs x nvars x d_model x num_patch]
#        output: [bs x forecast_len x nvars]
#        """
        
#        x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)] 
#        #print('flatten',x.shape)#torch.Size([64, 7, 3968])
#        x = self.dropout(x)
#        #x=self.gelu(x)
#        #print('dropout',x.shape)
#        x = x.view(32, 16*256) 
#        x = self.linear(x)      # x: [bs x nvars x forecast_len]
#        #print('linear',x.shape)
#        x = x.view(32, 96,7)
#        #print('view',x.shape)#torch.Size([64, 7, 96])
#        return x #.transpose(2,1)     # [bs x forecast_len x nvars]#torch.Size([64, 7, 96])


#class Head(nn.Module):
#    def __init__(self, dim, out_shape, p_drop=0.):
#        super(Head, self).__init__()
#        self.norm = nn.LayerNorm(dim)
#        self.activation = nn.GELU()
#        self.dropout = nn.Dropout(p_drop)
#        self.fc = nn.Linear(dim, out_shape)

#    def forward(self, x):
#        x = self.norm(x)
#        x = self.activation(x)
#        x = self.dropout(x)
#        x = self.fc(x)
#        print ( "thei is x" , x)


#        return x

#class Head(nn.Module):
#    def __init__(self, in_channels, d_model, output_dim, head_dropout, y_range=None):
#        super(Head, self).__init__()
#        self.y_range = y_range
#        self.flatten = nn.Flatten(start_dim=1)
#        self.dropout = nn.Dropout(head_dropout)
#        print(f"in_channels: {in_channels}, d_model: {d_model}, output_dim: {output_dim}")

#        #self.linear = nn.Linear(in_channels * d_model, output_dim)
#        #self.linear = nn.Linear(in_channels * d_model, 32*7)
#        self.linear = nn.Linear(64 * 256, 32*7)



#    def forward(self, x):
#        """
#        x: [bs x nvars x d_model x num_patch]
#        output: [bs x output_dim]
#        """
#        print(x.shape)

#        #x = x[:, :, :, -1]  # only consider the last item in the sequence, x: bs x nvars x d_model
#        x = x[:, :, -1]

#        x = self.flatten(x)  # x: bs x nvars * d_model
#        # Reshape x to match the expected shape for the linear layer
#        #x = x.reshape(32, 1792)

#        x = self.dropout(x)
#        y = self.linear(x)  # y: bs x output_dim
#        if self.y_range:
#            y = SigmoidRange(*self.y_range)(y)
#        return y




    #class GlobalAvgPool(nn.Module):
#    def forward(self, x):
#        return x.mean(dim=-2)

#class Head(nn.Sequential):
#    def __init__(self, dim, classes, p_drop=0.):
#        super().__init__(
#            nn.LayerNorm(dim),
#            nn.GELU(),
#            GlobalAvgPool(),
#            nn.Dropout(p_drop),
#            nn.Linear(dim, classes)
#        )


