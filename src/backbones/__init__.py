from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
from .swin import swinb
from .vit import vitb
from .resnet import r50, r101
from .mobilenetv3 import mobilenetv3s, mobilenetv3l
from .efficientnet import efficientnetb0, efficientnetv2s, efficientnetv2m, efficientnetv2l

def get_model(name, **kwargs):
    # resnet
    if name == "ir18":
        return iresnet18(False, **kwargs)
    elif name == "ir34":
        return iresnet34(False, **kwargs)
    elif name == "ir50":
        return iresnet50(False, **kwargs)
    elif name == "ir100":
        return iresnet100(False, **kwargs)
    elif name == "ir200":
        return iresnet200(False, **kwargs)
    elif name == 'swinb':
        return swinb()
    elif name == 'vitb':
        return vitb()
    elif name == 'r50':
        return r50()
    elif name == 'r101':
        return r101()
    elif name == 'mobilenetv3' or name == 'mobilenetv3l':
        return mobilenetv3l()
    elif name == 'mobilenetv3s':
        return mobilenetv3s()
    elif name == 'efficientnet' or name == 'efficientnetv2s':
        return efficientnetv2s()
    elif name == 'efficientnetb0':
        return efficientnetb0()
    elif name == 'efficientnetv2m':
        return efficientnetv2m()
    elif name == 'efficientnetv2l':
        return efficientnetv2l()
    else:
        raise ValueError()

