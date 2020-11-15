from lib.opts import opts
from lib.models.model import create_model, load_model
from types import MethodType
import torch.onnx as onnx
import torch
from torch.onnx import OperatorExportTypes
from collections import OrderedDict


INPUT_MODEL = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/ObjectDetection/CenterNet/CenterNet-train/exp/ctdet/coco_mobilenetv2_10_512/model_best.pth"
OUTPUT_MODEL = "/mnt/DATA/GRADUATION_RESEARCH/Workstation_Sources/ObjectDetection/CenterNet/trained_models/coco_mobilenetv2_512.onnx"
INPUT_SIZE = 512

def mobilenet_forward(self, x):
    x = self.base(x)
    x = self.dla_up(x)
    ret = []
    for head in self.heads:
        ret.append(self.__getattr__(head)(x))
    return ret

forward = {'mobilenetv2': mobilenet_forward}

opt = opts().init()
opt.task = 'ctdet'
opt.num_classes = 10
opt.arch = 'mobilenetv2_10'
opt.heads = OrderedDict([('hm', 10), ('reg', 2), ('wh', 2)])
opt.head_conv = 64
model = create_model(opt.arch, opt.heads, opt.head_conv)
model.forward = MethodType(forward[opt.arch.split('_')[0]], model)
load_model(model, INPUT_MODEL)
model.eval()
model.cuda()
input = torch.zeros([1, 3, INPUT_SIZE, INPUT_SIZE]).cuda()
onnx.export(model, input, OUTPUT_MODEL, verbose=True,
            operator_export_type=OperatorExportTypes.ONNX)