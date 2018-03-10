from torch import nn

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=32, stride=2, padding=15), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=16, stride=2, padding=7), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=8, stride=2, padding=3), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),  
        )

    def forward(self, x):
        x = self.main(x)
        return x

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=8, stride=2, padding=3),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=16, stride=2, padding=7), 
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=32, stride=2, padding=15), 
        )

    def forward(self, x):
        x = self.main(x)
        return x        

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.e = encoder()
        self.d = decoder()

    def forward(self, x):
        x = self.e(x)
        x = self.d(x)
        return x

