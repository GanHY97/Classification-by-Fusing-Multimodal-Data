import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self,  num_classes=2, init_weights=False):
        super(VGG, self).__init__()
        self.classifier = nn.Sequential(

            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)

        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):


        x = torch.flatten(x, start_dim=1)

        # N x 512*7*7
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

    def get_linear(self,x):

        x = torch.flatten(x, start_dim=1)
        self.classifier1 = nn.Sequential(

            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),

        )
        x=self.classifier1(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




def vgg(**kwargs):

    model = VGG(**kwargs)
    return model