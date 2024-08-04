import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder1 = self.conv_block(in_channels, base_channels)
        self.encoder2 = self.conv_block(base_channels, base_channels*2)
        self.encoder3 = self.conv_block(base_channels*2, base_channels*4)
        self.encoder4 = self.conv_block(base_channels*4, base_channels*8)
        
        # Bottleneck
        self.middle = self.conv_block(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = self.deconv_block(base_channels*16, base_channels*8)
        self.decoder4 = self.conv_block(base_channels*16, base_channels*8)
        
        self.upconv3 = self.deconv_block(base_channels*8, base_channels*4)
        self.decoder3 = self.conv_block(base_channels*8, base_channels*4)
        
        self.upconv2 = self.deconv_block(base_channels*4, base_channels*2)
        self.decoder2 = self.conv_block(base_channels*4, base_channels*2)
        
        self.upconv1 = self.deconv_block(base_channels*2, base_channels)
        self.decoder1 = self.conv_block(base_channels*2, base_channels)
        
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        #print("X",x.shape)
        enc1 = self.encoder1(x)
        #print("ENC1",enc1.shape)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        #print("ENC2",enc2.shape)
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        #print("ENC3",enc3.shape)
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        #print("ENC4",enc4.shape)
        
        # Bottleneck
        middle = self.middle(F.max_pool2d(enc4, 2))
        #print("MIDDLE",middle.shape)
        
        # Decoder with skip connections
        dec4 = self.upconv4(middle)
        #print("DEC4",dec4.shape)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        #print("DEC4SKIP",dec4.shape)
        dec4 = self.decoder4(dec4)
        #print("DEC4DECODE",dec4.shape)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)
        #print("FINAL", self.final(dec1).shape)
        return self.final(dec1)


class DiffusionModel(nn.Module):
    def __init__(self, unet_model, device, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.unet_model = unet_model
        self.device = device
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, axis=0).to(self.device)
        
    def forward_diffusion(self, x0, t):
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return xt, noise
    
    def reverse_diffusion(self, xt, t):
        predicted_noise = self.unet_model(xt)
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        xt_minus_1 = (xt - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        return xt_minus_1, predicted_noise

    def forward(self, x0, t):
        xt, noise = self.forward_diffusion(x0, t)
        xt_minus_1, predicted_noise = self.reverse_diffusion(xt, t)

        return xt_minus_1, predicted_noise, noise
    
    def loss_function(self, x0, t):

        loss = F.mse_loss()

        return 

    def sample(self, shape, device):

        xt = torch.randn(shape).to(device)
        with torch.no_grad():
            for t in reversed(range(self.num_timesteps)):
                xt, _ = self.reverse_diffusion(xt, torch.tensor([t]).to(device))
            return xt
