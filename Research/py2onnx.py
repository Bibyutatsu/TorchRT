"""Pytorch to ONNX conversion module

This module helps in the conversion of Pytorch models to ONNX for
faster inference

    Typical usage:
    Change the model_path, ONNX_path, input_shape and model
    > python py2onnx.py
"""

from shutil import copy

import torch
from networks import torNet, FashionCNN, efficientNet

import os
import onnx
from onnxsim import simplify


def load_model_weight(model, model_path):
    """Load the model using .pt/.pth file

    Loads the pytorch model using the PTH file containing the model parameters

    Args:
        model: The Pytorch model without trained weights
        model_path: The PTH/PT file location

    Returns:
        model: The updated pytorch model
    """

    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model


def export_onnx_model(model,
                      input_shape,
                      onnx_path,
                      input_names=None,
                      output_names=None,
                      dynamic_axes=None):
    """Export Pytorch model to ONNX

    Helps in creating the ONNX file from the pytorch model.
    Actual code by zong fan
    https://medium.com/@fanzongshaoxing/accelerate-pytorch-model-with-tensorrt-via-onnx-d5b5164b369

    Args:
        model: The pytorch model
        input_shape: Dummy input shape for the model
        onnx_path: The path where the ONNX file needs to be saved
        input_names: Optional, List containing input labels
        output_names: Optional,  List containing output labels
        dynamic_axis: Optional, Dictionary for variable batch_size
    """
    inputs = torch.ones(*input_shape)
    model(inputs)
    torch.onnx.export(model,
                      inputs,
                      onnx_path,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11)
    return


if __name__ == "__main__":
    model_path = "./../Models/efficientNet.pth"
    onnx_path = "./../Models/efficientNet.onnx"
    cpp_model_folder = './../Inference/data/fashionmnist/onnx'

    input_names = ['input']
    output_names = ['output']
    input_shape = (1, 1, 28, 28)
#     dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    model = efficientNet()
    model = load_model_weight(model, model_path)
    if os.path.basename(model_path).lower() == "efficientnet.pth":
        model.onnx_swish()

    try:
        export_onnx_model(model,
                          input_shape,
                          onnx_path,
                          input_names=input_names,
                          output_names=output_names)
#                           dynamic_axes=dynamic_axes)
        copy(onnx_path, cpp_model_folder)
    except Exception as e:
        print('Model Not converted due to :', e)
        raise

    try:
        # Simplify onnx model
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model)

        assert check, "Simplified ONNX model could not be validated"

        onnx.save(model_simp, onnx_path)
    except Exception as e:
        print('Model Not simplified due to :', e)

    copy(onnx_path, cpp_model_folder)
