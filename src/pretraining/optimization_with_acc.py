# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
from albert import lamb_optimizer
import six
from six.moves import zip
import tensorflow.compat.v1 as tf
from tensorflow.contrib import tpu as contrib_tpu


def _get_learning_rate(init_lr, num_train_steps, num_warmup_steps, poly_power=1.0, start_warmup_step=0):
  """Creates learning rate"""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
    learning_rate,
    global_step,
    num_train_steps,
    end_learning_rate=0.0,
    power=poly_power,
    cycle=False)

  # Add learning rate to tensorboard logs
  # tf.summary.scalar('learning_rate', learning_rate)

  # Implements linear warmup. I.e., if global_step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step)
                    + ", for " + str(num_warmup_steps) + " steps ++++++")
    global_steps_int = tf.cast(global_step, tf.int32)
    start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
    global_steps_int = global_steps_int - start_warm_int
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
            (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    return learning_rate


def _get_optimizer(learning_rate, use_tpu, optimizer="adamw"):
  """Creates an optimizer"""
  # It is OK that you use this optimizer for finetuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
  # As report in the Bert pulic github, the learning rate for SQuAD 1.1 finetune
  # is 3e-5, 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a
  # batch size of 64 in the finetune.
  if optimizer == "adamw":
    tf.logging.info("using adamw")
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer == "lamb":
    tf.logging.info("using lamb")
    optimizer = lamb_optimizer.LAMBOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    raise ValueError("Not supported optimizer: ", optimizer)

  if use_tpu:
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)
  return optimizer


def _get_train_op(loss, optimizer, num_accumulation_steps=1, colocate_gradients_with_ops=False):
  """Creates training op."""
  if num_accumulation_steps > 1:
    tf.logging.info("++++++ grad accumulation steps: " + str(num_accumulation_steps)
                    + " ++++++")

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # NVIDIA implementation adaptation
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  global_step = tf.train.get_global_step()
  tvars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(
      loss * 1.0 / num_accumulation_steps, tvars,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  if num_accumulation_steps > 1:
      local_step = tf.get_variable(name="local_step", shape=[], dtype=tf.int32, trainable=False,
                                   initializer=tf.zeros_initializer)
      batch_finite = tf.get_variable(name="batch_finite", shape=[], dtype=tf.bool, trainable=False,
                                     initializer=tf.ones_initializer)
      accum_vars = [tf.get_variable(
          name=tvar.name.split(":")[0] + "/accum",
          shape=tvar.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer()) for tvar in tf.trainable_variables()]

      reset_step = tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool)
      local_step = tf.cond(reset_step, lambda: local_step.assign(tf.ones_like(local_step)), lambda: local_step.assign_add(1))

      grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i]) for i, gv in enumerate(grads_and_vars) if gv[0] is not None]
      grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

      all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
      # if manual_fp16 or use_fp16 else tf.constant(True, dtype=tf.bool)
      batch_finite = tf.cond(reset_step,
        lambda: batch_finite.assign(tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
        lambda: batch_finite.assign(tf.math.logical_and(batch_finite, all_are_finite)))

      # This is how the model was pre-trained.
      # ensure global norm is a finite number
      # to prevent clip_by_global_norm from having a hizzy fit.
      (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=1.0,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(1.0)))

      accum_vars = tf.cond(reset_step,
              lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(clipped_grads)],
              lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(clipped_grads)])

      def update(accum_vars):
          return optimizer.apply_gradients(list(zip(accum_vars, tvars)), global_step=global_step)

      update_step = tf.identity(tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool), name="update_step")
      update_op = tf.cond(update_step,
                          lambda: update(accum_vars), lambda: tf.no_op())

      new_global_step = tf.cond(tf.math.logical_and(update_step, batch_finite),
                                lambda: global_step+1,
                                lambda: global_step)
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(update_op, [global_step.assign(new_global_step)])
  else:
      grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
      grads, tvars = list(zip(*grads_and_vars))
      all_are_finite = tf.constant(True, dtype=tf.bool)

      # This is how the model was pre-trained.
      # ensure global norm is a finite number
      # to prevent clip_by_global_norm from having a hizzy fit.
      (clipped_grads, _) = tf.clip_by_global_norm(
          grads, clip_norm=1.0,
          use_norm=tf.cond(
              all_are_finite,
              lambda: tf.global_norm(grads),
              lambda: tf.constant(1.0)))

      train_op = optimizer.apply_gradients(
          list(zip(clipped_grads, tvars)), global_step=global_step)

      new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     optimizer="adamw", poly_power=1.0, start_warmup_step=0,
                     num_accumulation_steps=1,
                     colocate_gradients_with_ops=False):
  """Creates an optimizer training op."""

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Deprecated: decomposed into _get_learning_rate, _get_optimizer and
  # _get_train_op for clarity and logging purposes.
  # Templates is inspired from:
  # https://github.com/tensorflow/tpu/blob/master/models/experimental/deeplab/model.py
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  global_step = tf.train.get_or_create_global_step()
  # tf.logging.info("++++++ global training step: " + str(global_step)
  #                 + " ++++++")

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=poly_power,
      cycle=False)

  # Add learning rate to tensorboard  logs
  # tf.summary.scalar('learning_rate', learning_rate)

  # Implements linear warmup. I.e., if global_step - start_warmup_step <
  # num_warmup_steps, the learning rate will be
  # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step)
                    + ", for " + str(num_warmup_steps) + " steps ++++++")
    global_steps_int = tf.cast(global_step, tf.int32)
    start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
    global_steps_int = global_steps_int - start_warm_int
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is OK that you use this optimizer for finetuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
  # As report in the Bert pulic github, the learning rate for SQuAD 1.1 finetune
  # is 3e-5, 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a
  # batch size of 64 in the finetune.
  if optimizer == "adamw":
    tf.logging.info("using adamw")
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer == "lamb":
    tf.logging.info("using lamb")
    optimizer = lamb_optimizer.LAMBOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  else:
    raise ValueError("Not supported optimizer: ", optimizer)

  if use_tpu:
    optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

  if num_accumulation_steps > 1:
    tf.logging.info("++++++ grad accumulation steps: " + str(num_accumulation_steps)
                    + " ++++++")

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # NVIDIA implementation adaptation
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  tvars = tf.trainable_variables()
  grads_and_vars = optimizer.compute_gradients(
      loss * 1.0 / num_accumulation_steps, tvars,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  if num_accumulation_steps > 1:
      local_step = tf.get_variable(name="local_step", shape=[], dtype=tf.int32, trainable=False,
                                   initializer=tf.zeros_initializer)
      batch_finite = tf.get_variable(name="batch_finite", shape=[], dtype=tf.bool, trainable=False,
                                     initializer=tf.ones_initializer)
      accum_vars = [tf.get_variable(
          name=tvar.name.split(":")[0] + "/accum",
          shape=tvar.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer()) for tvar in tf.trainable_variables()]

      reset_step = tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool)
      local_step = tf.cond(reset_step, lambda: local_step.assign(tf.ones_like(local_step)), lambda: local_step.assign_add(1))

      grads_and_vars_and_accums = [(gv[0], gv[1], accum_vars[i]) for i, gv in enumerate(grads_and_vars) if gv[0] is not None]
      grads, tvars, accum_vars = list(zip(*grads_and_vars_and_accums))

      all_are_finite = tf.reduce_all([tf.reduce_all(tf.is_finite(g)) for g in grads])
      # if manual_fp16 or use_fp16 else tf.constant(True, dtype=tf.bool)
      batch_finite = tf.cond(reset_step,
        lambda: batch_finite.assign(tf.math.logical_and(tf.constant(True, dtype=tf.bool), all_are_finite)),
        lambda:batch_finite.assign(tf.math.logical_and(batch_finite, all_are_finite)))

      # This is how the model was pre-trained.
      # ensure global norm is a finite number
      # to prevent clip_by_global_norm from having a hizzy fit.
      (clipped_grads, _) = tf.clip_by_global_norm(
            grads, clip_norm=1.0,
            use_norm=tf.cond(
                all_are_finite,
                lambda: tf.global_norm(grads),
                lambda: tf.constant(1.0)))

      accum_vars = tf.cond(reset_step,
              lambda: [accum_vars[i].assign(grad) for i, grad in enumerate(clipped_grads)],
              lambda: [accum_vars[i].assign_add(grad) for i, grad in enumerate(clipped_grads)])

      def update(accum_vars):
          return optimizer.apply_gradients(list(zip(accum_vars, tvars)), global_step=global_step)

      update_step = tf.identity(tf.cast(tf.math.equal(local_step % num_accumulation_steps, 0), dtype=tf.bool), name="update_step")
      update_op = tf.cond(update_step,
                          lambda: update(accum_vars), lambda: tf.no_op())

      new_global_step = tf.cond(tf.math.logical_and(update_step, batch_finite),
                                lambda: global_step+1,
                                lambda: global_step)
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(update_op, [global_step.assign(new_global_step)])
  else:
      grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
      grads, tvars = list(zip(*grads_and_vars))
      all_are_finite = tf.constant(True, dtype=tf.bool)

      # This is how the model was pre-trained.
      # ensure global norm is a finite number
      # to prevent clip_by_global_norm from having a hizzy fit.
      (clipped_grads, _) = tf.clip_by_global_norm(
          grads, clip_norm=1.0,
          use_norm=tf.cond(
              all_are_finite,
              lambda: tf.global_norm(grads),
              lambda: tf.constant(1.0)))

      train_op = optimizer.apply_gradients(
          list(zip(clipped_grads, tvars)), global_step=global_step)

      new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=six.ensure_str(param_name) + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=six.ensure_str(param_name) + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", six.ensure_str(param_name))
    if m is not None:
      param_name = m.group(1)
    return param_name

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Je sais plus à quoi correspond cette implementation ...
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # accum_steps_const = tf.constant(num_accumulation_steps, dtype=tf.int32)
  # accum_steps_count = tf.Variable(0, dtype=tf.int32, trainable=False)
  #
  # tvars = tf.trainable_variables()
  # accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvars]
  #
  # grads = tf.gradients(loss, tvars)
  #
  # accum = [accum_vars[i].assign_add(gv / tf.cast(accum_steps_const, tf.float32), use_locking=True) for i, gv in
  #          enumerate(grads)]
  # accum_ops = tf.group(accum, accum_steps_count.assign(accum_steps_count + 1, use_locking=True))
  #
  # def grad_step():
  #     zero_ops = tf.group([tv.assign(tf.zeros_like(tv), use_locking=True) for tv in accum_vars])
  #     with tf.control_dependencies(
  #             [accum_ops, tf.Print(accum_steps_const, [accum_steps_const], "UPDATING GRADIENTS: ")]):
  #         (new_accum_vars, _) = tf.clip_by_global_norm(accum_vars, clip_norm=1.0)
  #         apply_grads = optimizer.apply_gradients(zip(new_accum_vars, tvars), global_step=global_step)
  #         with tf.control_dependencies([apply_grads]):
  #             train_op = tf.group(zero_ops, accum_steps_count.assign(0, use_locking=True),
  #                                 global_step.assign(global_step + 1, use_locking=True))
  #     return train_op
  #
  # def accum_step():
  #     return accum_ops
  #
  # # This is how the model was pre-trained.
  # out_op = tf.cond(accum_steps_count >= (accum_steps_const - 1), grad_step, accum_step)
  #
  # # Normally the global step update is done inside of `apply_gradients`.
  # # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # # a different optimizer, you should probably take this line out.
  # return out_op

  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # Albert implementation
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # tvars = tf.trainable_variables()
  # grads = tf.gradients(
  #     loss, tvars, colocate_gradients_with_ops=colocate_gradients_with_ops)
  #
  # # This is how the model was pre-trained.
  # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  #
  # train_op = optimizer.apply_gradients(
  #     list(zip(grads, tvars)), global_step=global_step)
  #
  # # Normally the global step update is done inside of `apply_gradients`.
  # # However, neither `AdamWeightDecayOptimizer` nor `LAMBOptimizer` do this.
  # # But if you use a different optimizer, you should probably take this line
  # # out.
  # new_global_step = global_step + 1
  # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  # return train_op