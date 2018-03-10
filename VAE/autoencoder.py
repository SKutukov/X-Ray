from torch import nn

class autoencoder(nn.Module):
    
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=16, stride=2, padding=7), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=8, stride=2, padding=3), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=2, stride=2, padding=0),  
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=8, stride=2, padding=3),  # b, 1, 28, 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=16, stride=2, padding=7),  # b, 1, 28, 28
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

