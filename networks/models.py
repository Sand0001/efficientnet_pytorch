import torch.nn as nn

import timm
import config
from efficientnet_pytorch import EfficientNet
# dimdict={'efficientnet-b0':1536 ,'efficientnet-b1':1536 ,'efficientnet-b2':1536 ,'efficientnet-b3':1536 ,'efficientnet-b4':1792,'efficientnet-b5':2048,'efficientnet-b6':2304,'efficientnet-b7':2560}
dimdict={'efficientnet-b0':1536 ,'efficientnet-b1':1536 ,'efficientnet-b2':1408 ,'efficientnet-b3':1536 ,'efficientnet-b4':1792,'efficientnet-b5':2048,'efficientnet-b6':2304,'efficientnet-b7':2560}

class terrorismNet(nn.Module):
    def __init__(self,TIMM_MODEL):
        super().__init__()
        # TIMM_MODEL = 'efficientnet_b3a'
        self.TIMM_MODEL=TIMM_MODEL
        if 'efficientnet-' in self.TIMM_MODEL:


            self.backbone = EfficientNet.from_pretrained(self.TIMM_MODEL)
            self.n_features = dimdict[self.TIMM_MODEL]

        elif 'resnext50_32x4d' == self.TIMM_MODEL or 'tf_efficientnet_b3_ns' == self.TIMM_MODEL or 'tf_efficientnet_b2_ns' == self.TIMM_MODEL :

            self.backbone = timm.create_model(self.TIMM_MODEL, pretrained=True)
            print(self.backbone)
            if 'efficient' in self.TIMM_MODEL:
                self.n_features = self.backbone.classifier.in_features
            else:
                self.n_features = self.backbone.fc.in_features
            self.backbone = nn.Sequential(*self.backbone.children())[:-2]
            print(self.backbone)

            self.classifier = nn.Linear(self.n_features, config.class_num)
            # self.backbone.fc = nn.Linear(self.n_features, config.class_num)

        else:

            backbone = timm.create_model(self.TIMM_MODEL, pretrained=True)
            self.n_features = backbone.fc.in_features#提取fc层中的固定的参数
            self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(self.n_features,config.class_num )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        if 'efficientnet-' in self.TIMM_MODEL:
            x = self.backbone.extract_features(x)
            fea_pool = self.pool(x).view(x.size(0), -1)
            logits = self.classifier(fea_pool)
            return logits, x.detach()
        elif 'resnext50_32x4d' == self.TIMM_MODEL or 'tf_efficientnet_b3_ns' == self.TIMM_MODEL or 'tf_efficientnet_b2_ns' == self.TIMM_MODEL:
            feats= self.backbone(x)
            x = self.pool(feats).view(x.size(0), -1)
            x = self.classifier(x)
            return x,feats



        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats

class ENet(nn.Module):

    def __init__(self,conf):
        super(ENet, self).__init__()
        self.basemodel = EfficientNet.from_pretrained(conf.netname)
        feadim=dimdict[conf.netname]
        self.classifier = nn.Linear(feadim, conf.num_class)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.basemodel.extract_features(x)
        fea_pool = self.avg_pool(x).view(x.size(0), -1)
        logits = self.classifier(fea_pool)
        return logits,x.detach()


    # def get_params(self, param_name):
    #     ftlayer_params = list(self.basemodel.parameters())
    #     ftlayer_params_ids = list(map(id, ftlayer_params))
    #     freshlayer_params = filter(lambda p: id(p) not in ftlayer_params_ids, self.parameters())
    #
    #     return eval(param_name+'_params')

class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x