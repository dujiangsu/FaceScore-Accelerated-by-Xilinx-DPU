# inpect_checkpoint.py
"""
A simple script for inspect checkpoint files.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("file_name", "", "Checkpoint filename")
tf.app.flags.DEFINE_string("tensor_name", "", 
                           "Name of the tensor to inspect")

def print_tensors_in_checkpoint_file(file_name, tensor_name):
  """
  打印 checkpoint 文件中的 tensors.
  
 如果未指定 `tensor_name`，则打印 checkpoint 文件中的 tensor names 和 shapes.
 如果指定了 `tensor_name`，则打印该 tensor 的内容.
 Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
  """
  try:
    reader = tf.train.NewCheckpointReader(file_name)
    if not tensor_name:
      print(reader.debug_string().decode("utf-8"))
    else:
      print("tensor_name: ", tensor_name)
      #print(reader.get_tensor(tensor_name))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed with SNAPPY.")


def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint "
          "--file_name=checkpoint_file_name "
          "[--tensor_name=tensor_to_print]")
    sys.exit(1)
  else:
    print_tensors_in_checkpoint_file(FLAGS.file_name, 
                                     FLAGS.tensor_name)



checkpoint_file = './chkpts/float_model.ckpt'

reader = tf.train.NewCheckpointReader(checkpoint_file)
#print(reader.debug_string().decode("utf-8"))
var_to_shape_map = reader.get_variable_to_shape_map() 
for key in var_to_shape_map: 
    print("tensor_name: ", key)   # 打印变量名
   # print(reader.get_tensor(key)) # 打印变量值 
