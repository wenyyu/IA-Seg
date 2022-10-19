import torch.nn as nn

class FCDiscriminator(nn.Module):
	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=1, padding=1)
		self.conv4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=1, padding=1)
		self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=1)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		return x
# DISCRIMINATOR NETWORK
class Discriminator(nn.Module):
    def __init__(self,num_classes):
        super(Discriminator, self).__init__()

        #  Convolutional layers
        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_classes, 16, 5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(16)
        )

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(32)
        )

        # input 256x256x32  output 128x128x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(64)
        )

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm2d(128)
        )

        # input 16x16x128  output 1x1x128
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 1, 16),
            nn.LeakyReLU(inplace=True)
        )

        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1(x)

        # input 512x512x16 to output 256x256x32
        x = self.conv2(x)

        # input 256x256x32 to output 128x128x64
        x = self.conv3(x)

        # input 128x128x64 to output 64x64x128
        x = self.conv4(x)

        # input 64x64x128 to output 32x32x128
        x = self.conv5(x)

        # input 32x32x128 to output 16x16x128
        x = self.conv6(x)

        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)
        # print(x.shape)

        x = self.fc(x)

        return x

