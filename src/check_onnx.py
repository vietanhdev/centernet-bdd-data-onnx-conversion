import torch.onnx as onnx

model = onnx.load("model_best_vtpost_dla.onnx")

onnx.checker.check_model(model)


onnx.helper.printable_graph(model.graph)


print("OK!")