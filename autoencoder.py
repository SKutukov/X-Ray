from torch import nn

class autoencoder(nn.Module):
    
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=64, stride=2, padding=31), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=16, stride=2, padding=7), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=8, stride=2, padding=3), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, padding=0),  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=2, stride=2, padding=0),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=8, stride=2, padding=3),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=16, stride=2, padding=7), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=64, stride=2, padding=31), 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

