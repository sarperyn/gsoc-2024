import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(ConvVAE, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # (32, 133, 133)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # (64, 67, 67)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # (128, 34, 34)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # (256, 17, 17)
        self.enc_fc1 = nn.Linear(256 * 16 * 16, 512)
        self.enc_fc2_mu = nn.Linear(512, latent_dim)
        self.enc_fc2_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, 256 * 16 * 16)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # (128, 34, 34)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0) # (64, 67, 67)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # (32, 133, 133)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=0) # (1, 266, 266)
    
    def encode(self, x):
        h1 = F.relu(self.enc_conv1(x))
        h2 = F.relu(self.enc_conv2(h1))
        h3 = F.relu(self.enc_conv3(h2))
        h4 = F.relu(self.enc_conv4(h3))
        h4 = h4.view(h4.size(0), -1)
        h5 = F.relu(self.enc_fc1(h4))
        return self.enc_fc2_mu(h5), self.enc_fc2_logvar(h5)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h1 = F.relu(self.dec_fc1(z))
        h2 = F.relu(self.dec_fc2(h1))
        h2 = h2.view(h2.size(0), 256, 16, 16)
        h3 = F.relu(self.dec_conv1(h2))
        h4 = F.relu(self.dec_conv2(h3))
        h5 = F.relu(self.dec_conv3(h4))
        return torch.sigmoid(self.dec_conv4(h5))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


def calculate_conv_dims(input_size,paddings:list,kernels:list,strides:list,maxpool:list):
    
    outputs = []
    outputs.append(input_size)
    for i in range(len(paddings)):
        
        output_size = (input_size + (2*paddings[i]) - (kernels[i] - 1) - 1)/strides[i] + 1
        if maxpool[i] != 0:
            output_size = (output_size  + (2*paddings[i]) - (maxpool[i]-1)-1)/2 +1
        
        outputs.append(int(output_size))
        input_size = output_size
        
    print(outputs)
    return outputs


def calculate_convtrans_dim(input_size,paddings:list,kernels:list,strides:list):
    outputs = []
    outputs.append(input_size)
    for i in range(len(paddings)):
        
        output_size = (input_size - 1) * strides[i]  -  2 * paddings[i] + kernels[i] - 1 + 1
        outputs.append(int(output_size))
        input_size = output_size
        
    print(outputs)
    return outputs


if __name__ == '__main__':

    # kernels_enc = [4,4,4,4]
    # paddings_enc= [1,1,1,1]
    # strides_enc = [2,2,2,2]

    # maxpool = [0,0,0,0,0]

    # kernels_dec  = [4,4,4,4]
    # paddings_dec = [1,0,1,0]
    # strides_dec  = [2,2,2,2]


    # convdim_outputs = calculate_conv_dims(266,paddings_enc,kernels_enc,strides_enc,maxpool)
    # convtrans_outputs = calculate_convtrans_dim(16,paddings_dec,kernels_dec,strides_dec)

    input_ = torch.randn(5, 1, 266, 266)
    model  = ConvVAE()
    out    = model(input_)
    print(out[0].shape)