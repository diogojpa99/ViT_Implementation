import torch
import torch.nn as nn



class PatchEmbed(nn.Module):    
    ''' 
    Split the image into patches and then embed them.

        Parameters:
        -----------
        1) img_size : int 
            Size of the image (it is a square) [Not mandatory].
              
        2) patch_size : int
            Size of the patch (it is a square).
        
        3) in_chans : int
            Number of input channels.
        
        4) embed_dim : int
            The embedding dimension.
            
        Atributes:
        ----------
        1) n_patches : int
            Number of patches inside of our image
            
        2) proj : nn.Conv2d
            Convolutional layer that does both the splitting into patches
            and their embedding.
    '''
    # Notes:
    # This embedding will remain constant trought the entire network.
    
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size)** 2 #Is this correct ? Yes.
        # This is right, but only assuming that our images are squares
                                                    
        
        # proj:
        # It is a 2D Convolutional layer that split the image into patches
        # and gives us their respective embeddings
        # If you notice the kernel_size and stride are both equal to the patch_size
        # This way when we are sliding the kernel along the input tensor we will never
        # Slide in an overlaping way
        self.proj = nn.Conv2d( in_chans,embed_dim, kernel_size=patch_size, stride =patch_size)
        
    def forward(self, x):
        '''
        Run the forward pass.
        
        Parameters:
        -----------
        1) x : torch.Tensor
            Shape '(n_samples, in_chans, img_size, img_size)'
            
        Returns:
        --------
        1) torch.Tensor
            Shape '(n_samples, n_patches, embed_dim)'.
            [This is what goes inside the tranformer layer - 3D Tensor]
                         
        '''
        
        x = self.proj(x) #(n_samples, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        # Shouldn't the flatten process be before the projection ?? Maybe not...
        x = x.flatten(2) #(n_samples, embed_dim, n_patches)
        x = x.transpose(1,2) #(n_samples, n_patches, embed_dim)
        
        # Note:
        # Where is the positional encoding ?????
        
        return x
    
class Attention(nn.Module):
    '''
    Attention mechanism.
    
    Parameters:
    ----------
    
    dim : int
        The input and output dimension per token features.
        
    n_heads : int
        Number of attention heads.
        
    qkv_bias : bool
        If True then we include bias to the query, key and value projections
        
    attn_p : float
        Dropout probability applied to the Q, K and V tensors
    
    proj_p : float
        Dropout probability applied to the output tensor 
        
    Attributes:
    -----------
    
    scale : float
        Normalizing constant for the dot product
    
    qkv : nn.Linear
        Linear projection for the Q, K and V
    
    proj : nn.Linear
        Linear mapping that takes in concatenated output of all attention heads
        and maps it into a new space
        
    attn_drop, proj_drop : nn.Dropout
        Dropout layers
        
    '''
    
    def __init__(self, dim, n_heads=12, qkv_bias = True, attn_p = 0., proj_p = 0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Note:
        # I think that this implementation consideres Q=K=V.
        # If this is the case, I don't think this is totally correct
        # Okay, maybe not!
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias) 
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self, x):
        '''
        Run the forward pass
        
        Parameters:
        -----------
        x: torch.Tensor
            Shape '(n_samples, n_patches +1, dim)'
            
        Returns:
        --------
        
        torch.Tensor
            Shape '(n_samples, n_patches + 1, dim)'
        
        '''
        
        n_samples, n_tokens, dim = x.shape
        
        if dim != self.dim:
            raise ValueError
        
        # I will try to connect the code to the theory
        # (1) We have to build the matrixes U_qkv
        qkv = self.qkv(x) #(n_samples, n_patches, +1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        ) #(n_samples, n_patches +1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2,0,3,1,4
        ) #(3, n_samples, n_heads, n_patches +1, head_dim)
        
        # (2) Compute Q,K and V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # (3) Self-Attention operations
        k_t = k.transpose(-2,-1) #(n_samples,n_heads, head_dim, n_patches +1)
        dp = (
            q @ k_t
        ) *self.scale #(n_samples, n_heads, n_patches +1, n_patches +1)
        attn = dp.softmax(dim = -1) #(n_samples, n_heads, n_patches +1, n_patches +1)
        attn = self.attn_drop(attn)
        
        # (4) Multiple Attention-Heads
        weighted_avg = attn @ v #(n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        ) #(n_samples, n_patches +1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) #(n_samples, n_patches +1, dim)
        
        x = self.proj(weighted_avg) #(n_samples, n_patches +1, dim)
        x = self.proj_drop(x) #(n_samples, n_patches +1, dim)
        
        return x
    
class MLP(nn.Module):
    '''
    Multilayer perceptron
    
    Parameters:
    ----------
    
    in_features : int
        Number of input features.
        
    hidden_features: int
        Number of nodes in the hidden layer.
        
    out_features: int
        Number of output features.
        
    p: float
        Dropout probability
        
    Attribute:
    ----------
    fc: nn.Linear
        The first Linear layer.
        
    act: nn.GELU
        GELU activation function.
        
    fc2: nn.Linear
        The second Linear layer
        
    drop: nn.Dropout
        Dropout layer
    '''
    
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        '''
        Run forward pass.
        
        Parameters:
        -----------
        
        x = torch.Tensor
            Shape '(n_samples, n_patches +1, in_features)'
            
        Returns:
        --------
        
        torch.Tensor
            Shape '(n_samples, n_patches +1, out_features)'
        '''
        
        x = self.fc1(x) #(n_samples, n_patches +1, hidden_features)
        x = self.act(x) #(n_samples, n_patches +1, hidden_features)
        x = self.fc2(x) #(n_samples, n_patches +1, hidden_features)
        x = self.drop(x) #(n_samples, n_patches +1, hidden_features)
        
        return x
    
# We have everything we need, now it is time to start putting things together
class Block(nn.Module):
    '''
    Transformer encoder block.
    
    Parameters:
    -----------
    
    dim : int
        Embedding dimension.
        
    n_heads : int
        Number of attention heads.
        
    mlp_ratio : float
        Determines the hidden dimension size of the 'MLP' module with respect to 'dim'
        
    qkv_bias: bool
        If True then we include bias to the q,k and v projections
        
    p, attn_p: float
        Dropout probability.
        
    Attributes:
    -----------
    
    norm1, norm2: LayerNorm
        Layer normalizaion
        
    attn: Attention
        Attention module
        
    mlp: MLP
        MLP module
    '''
    
    def __init__(self, dim, n_heads, mlp_ration=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2= nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ration)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim
        )
        
    def forward(self, x):
        '''
        Run forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            Shape '(n_samples, n_patches +1, dim)'
            
        Returns:
        -------
        torch.Tensor
            Shape '(n_samples, n_patches+1, dim)'
        '''
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
class VisionTransformer(nn.Module):
    '''
    Simplified implementation of the Vision Transformer
    
    Parameters:
    -----------
    
    img_size : int
        Both height and the width of the image (it is a square)
        
    patch_size : int
        Both height and the width of the patch (it is a square)
    
    in_chans : int
        Number of input channels
    
    n_classes : int
        Number of classes
        
    embed_dim : int
        Dimensionality of token/patch embeddings.
        
    depth : int
        Number of transformer encoder blocks.
    
    n_heads : int
        Number of attention heads.
        
    mlp_ratio : float
        Determines the hidden dimension of the 'MLP' module
        
    qkv_bias : bool
        If true then we include bias to the Q, V and K projections.
        
    p, attn_p : float
        Dropout probability.
        
    Attributes:
    -----------
    patch_embed : PatchEmbed
        Instance of 'PatchEmbed' layer.
    
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has 'embed_dim' elements.
        
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has '(n_patches + 1)' * embed_dim' elements.
        
    pos_drop : nn.Dropout
        Dropout layer.
        
    blocks : nn.ModuleList
        List of 'Block' modules.
        
    norm : nn.LayerNorm
        Layer normalization.
    '''
    
    def __init__(
        self,
        img_size=384, #Should I change this ?
        patch_size=16,
        in_chans=3,
        n_classes=1000, #Binary classification, shouldn't this be two ?
        embed_dim=768,
        depth =12,
        n_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        p=0.,
        attn_p=0.,  
    ):
        super().__init__()
        
        # (1) Patch embedding will be the first layer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        
        # (2) Define the cls token and initialize it with zeros
        self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim))
        
        # (3) Add the positional encoding parameter
        '''
        ! Attention !
        
        I need to do two types of experiments:
        
        (1) ViT with positional embeddings
        
        (2) ViT without positional embaddings.
            How do I do this ?
            So I need to get a pretrained model that was trained with positional embeddings
            But then in the fine-tuning I need to discardthem completely... 
            How do I do this ??
            I looks like putting all the positional embeddings as zero won't work, because 
            they are learnable positional embeddings... So How do I solve this ?
        '''
        self.pos_embed = nn.Parameter(
            torch.zeros(1,1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        
        # Here we iteratively create the transformer encoder
        # Notice that the hyperparameters of each block are the same
        # However each of the blocks will have your own learnable parameters
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ration=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        '''
        Run the forward pass.
        
        Parameters:
        -----------
        
        x : torch.Tensor
            Shape '(n_samples, in_chans, img_size, img_size)'.
        
        Returns:
        --------
        logits : torch.Tensor
            Logist over all the classes - '(n_samples, n_classes)'.
        '''
        # The input tensor is a batch of images
        n_samples = x.shape[0]
        # We take the input images and we turn them into patch embeddings
        x = self.patch_embed(x)
        
        # Concat the class token into the patch embedding
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        ) # (n_samples, 1, embed_dim)
        
        x = torch.cat((cls_token,x), dim=1) # (n_samples, n_patches + 1, embed_dim)
        
        # Add the positional embeddings 
        x = x + self.pos_embed #(n_samples, n_patches +1, embed_dim)
        
        # Apply a dropout
        x = self.pos_drop(x)
        
        # Define al the bocks of the transformer encoder
        for block in self.block:
            x = block(x)
            
        # Normalization layer
        x = self.norm(x)
        
        cls_token_final = x[:,0] # just the CLS token
        x = self.head(cls_token_final)
        
        return x
    
    
    
'''

I now know how to implement a ViT, but how to I fine-tune the model ?!

So basically this was not what I wanted

'''