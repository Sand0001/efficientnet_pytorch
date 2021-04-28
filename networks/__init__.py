from .models import terrorismNet
from .loss import ClassifyLoss,SnapMixLoss
from torch import nn
def get_model(model_name):
    return terrorismNet(TIMM_MODEL = model_name)

def get_loss(tag = None,training = False):
    if tag:
        criterion = SnapMixLoss()
    else:
        if training:
            criterion = ClassifyLoss(training=True)
        else:
            criterion = ClassifyLoss()

    return criterion
