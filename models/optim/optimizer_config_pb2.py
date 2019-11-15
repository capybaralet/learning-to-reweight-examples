# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: models/optim/optimizer_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='models/optim/optimizer_config.proto',
  package='models.optim',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n#models/optim/optimizer_config.proto\x12\x0cmodels.optim\"\xe3\x01\n\x0fOptimizerConfig\x12\x16\n\x04type\x18\x01 \x01(\t:\x08momentum\x12\x12\n\nlearn_rate\x18\x02 \x01(\x02\x12\x15\n\x08momentum\x18\x03 \x01(\x02:\x03\x30.9\x12\x12\n\nbatch_size\x18\x04 \x01(\x05\x12\x16\n\x0emax_train_iter\x18\x05 \x01(\x05\x12(\n\x19learn_rate_scheduler_type\x18\x06 \x01(\t:\x05\x66ixed\x12\x1e\n\x16learn_rate_decay_steps\x18\x07 \x03(\x05\x12\x17\n\x0flearn_rate_list\x18\x08 \x03(\x02')
)




_OPTIMIZERCONFIG = _descriptor.Descriptor(
  name='OptimizerConfig',
  full_name='models.optim.OptimizerConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='models.optim.OptimizerConfig.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("momentum").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learn_rate', full_name='models.optim.OptimizerConfig.learn_rate', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='momentum', full_name='models.optim.OptimizerConfig.momentum', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.9),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='models.optim.OptimizerConfig.batch_size', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_train_iter', full_name='models.optim.OptimizerConfig.max_train_iter', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learn_rate_scheduler_type', full_name='models.optim.OptimizerConfig.learn_rate_scheduler_type', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=True, default_value=_b("fixed").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learn_rate_decay_steps', full_name='models.optim.OptimizerConfig.learn_rate_decay_steps', index=6,
      number=7, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='learn_rate_list', full_name='models.optim.OptimizerConfig.learn_rate_list', index=7,
      number=8, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=54,
  serialized_end=281,
)

DESCRIPTOR.message_types_by_name['OptimizerConfig'] = _OPTIMIZERCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

OptimizerConfig = _reflection.GeneratedProtocolMessageType('OptimizerConfig', (_message.Message,), {
  'DESCRIPTOR' : _OPTIMIZERCONFIG,
  '__module__' : 'models.optim.optimizer_config_pb2'
  # @@protoc_insertion_point(class_scope:models.optim.OptimizerConfig)
  })
_sym_db.RegisterMessage(OptimizerConfig)


# @@protoc_insertion_point(module_scope)
