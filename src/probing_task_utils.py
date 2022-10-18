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
"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function
import collections
import csv
import os
import numpy as np
import pandas as pd
from albert import (
  fine_tuning_utils,
  modeling,
  optimization,
  tokenization)
import tensorflow.compat.v1 as tf
from tensorflow.contrib import (
  data as contrib_data,
  metrics as contrib_metrics,
  tpu as contrib_tpu)

from albert.classifier_utils import (
  # InputExample,
  PaddingInputExample,
  InputFeatures,
  # convert_single_example,
  # file_based_convert_examples_to_features,
  # file_based_input_fn_builder,
  # _truncate_seq_pair,
  # convert_examples_to_features,
  # create_model,
  # model_fn_builder,
  # input_fn_builder
)


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid,
               text_a,
               n_intervening=None,
               distance=None,
               words_to_mask_idx=None,
               pos_target=None,
               text_b=None,
               label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.n_intervening = n_intervening
    self.distance = distance
    self.words_to_mask_idx = words_to_mask_idx
    self.pos_target = pos_target
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""
  def __init__(self, use_spm, do_lower_case):
    super(DataProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case
    self.labels = None

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()\

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
    return lines

  @classmethod
  def _read_df(cls, input_file):
    """Reads a tab separated value file and store into pandas dataframe."""
    dataset = pd.read_csv(input_file, sep=',')
    return dataset

  def process_text(self, text):
    if self.use_spm:
      return tokenization.preprocess_text(text, lower=self.do_lower_case)
    else:
      return tokenization.convert_to_unicode(text)


class ProbingProcessor(DataProcessor):
  """Processor for the Length prediction data set (SentEval version)."""


  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, self.FILE_NAME)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, self.FILE_NAME)), "valid")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, self.FILE_NAME)), 'test')

  def get_labels(self):
    """See base class."""
    return self.labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""

    tok_2_split = {'tr': 'train', 'va': 'valid', 'te': 'test'}
    if self.labels is None:
      self.labels = sorted(np.unique([l[1] for l in lines]))

    examples = []
    for (i, line) in enumerate(lines):
      split = tok_2_split[line[0]]
      if (split != set_type) and (set_type is not None):
        continue
      text = self.process_text(line[2])
      label = line[1]
      examples.append(
        InputExample(guid=i, text_a=text, label=label))
    return examples


class ProbingMaskProcessor(ProbingProcessor):
  """Processor for the Length prediction data set (SentEval version)."""

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""

    tok_2_split = {'tr': 'train', 'va': 'valid', 'te': 'test'}
    if self.labels is None:
      self.labels = sorted(np.unique([l[1] for l in lines]))

    examples = []
    for (i, line) in enumerate(lines):
      split = tok_2_split[line[0]]
      if (split != set_type) and (set_type is not None):
        continue
      text = self.process_text(line[2])
      label = line[1]
      words_to_mask_idx = [int(line[3]) + 1]  # +1 for CLS
      examples.append(
        InputExample(guid=i, text_a=text, words_to_mask_idx=words_to_mask_idx, label=label))
    return examples


class ProbingDFProcessor(DataProcessor):
  """Processor for the Length prediction data set (SentEval version)."""


  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_df(os.path.join(data_dir, self.FILE_NAME)), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_df(os.path.join(data_dir, self.FILE_NAME)), "valid")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_df(os.path.join(data_dir, self.FILE_NAME)), 'test')

  def get_labels(self):
    """See base class."""
    return self.labels

  def _create_examples(self, df, set_type):
    """Creates examples for the training and dev sets."""

    tok_2_split = {'tr': 'train', 'va': 'valid', 'te': 'test'}
    df = df[df.columns.intersection(self.COLUMN_TO_KEEP)]
    if self.labels is None:
      self.labels = sorted(df[self.TARGET_COLUMN].unique())

    examples = []
    for i, line in df.iterrows():
      words_to_mask_idx = []
      split = tok_2_split[line['split']]
      if (split != set_type) and (set_type is not None):
        continue
      text = self.process_text(line['orig_sentence'])
      label = line[self.TARGET_COLUMN]
      n_intervening_nouns = int(line['n_intervening'])
      distance = int(line['distance'])
      if 'verb_index' in df.columns:
        words_to_mask_idx.append(int(line['verb_index']))
      for c in df.columns:
        if c.startswith('POS_TARGET_IDX'):
          words_to_mask_idx.append(int(line[c]) + 1)
      if 'POS_TARGET' in df.columns:
        pos_target = line['POS_TARGET']
      else:
        pos_target = None

      examples.append(
        InputExample(guid=i,
                     text_a=text,
                     label=label,
                     n_intervening=[n_intervening_nouns],
                     distance=[distance],
                     words_to_mask_idx=words_to_mask_idx,
                     pos_target=pos_target))
    return examples


class SubjectVerb(ProbingDFProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "linzen.txt"
  COLUMN_TO_KEEP = ['split', 'orig_sentence', 'n_intervening', 'distance', 'verb_pos', 'verb_index']
  TARGET_COLUMN = 'verb_pos'

class SubjectVerbMask1(ProbingDFProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "linzen_random_pos.txt"
  COLUMN_TO_KEEP = ['split', 'orig_sentence', 'n_intervening', 'distance', 'verb_pos', 'POS_TARGET_IDX', 'POS_TARGET']
  TARGET_COLUMN = 'POS_TARGET'

class SubjectVerbMask2(ProbingDFProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "linzen_random_pos_2.txt"
  COLUMN_TO_KEEP = ['split', 'orig_sentence', 'n_intervening', 'distance', 'verb_pos', 'POS_TARGET_IDX', 'POS_TARGET',
                    'POS_TARGET_IDX_2', 'POS_TARGET_2', 'POS_TARGET_ALL']
  TARGET_COLUMN = 'POS_TARGET_ALL'

class SubjectVerbMask3(ProbingDFProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "linzen_random_pos_3.txt"
  COLUMN_TO_KEEP = ['split', 'orig_sentence', 'n_intervening', 'distance', 'verb_pos', 'POS_TARGET_IDX', 'POS_TARGET',
                    'POS_TARGET_IDX_2', 'POS_TARGET_2', 'POS_TARGET_IDX_3', 'POS_TARGET_3', 'POS_TARGET_ALL']
  TARGET_COLUMN = 'POS_TARGET_ALL'

class SubjectVerbMask4(ProbingDFProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "linzen_random_pos_4.txt"
  COLUMN_TO_KEEP = ['split', 'orig_sentence', 'n_intervening', 'distance', 'verb_pos', 'POS_TARGET_IDX', 'POS_TARGET',
                    'POS_TARGET_IDX_2', 'POS_TARGET_2', 'POS_TARGET_IDX_3', 'POS_TARGET_3',
                    'POS_TARGET_IDX_3', 'POS_TARGET_4']
  TARGET_COLUMN = 'POS_TARGET'

class TopPosProcessor(ProbingProcessor):
  """Processor for the Adverb Type data set (ASI version)."""
  FILE_NAME = "top_dep.txt"

class AdverbMaskTypeProcessor(ProbingMaskProcessor):
  """Processor for the Adverb Type data set (ASI version)."""
  FILE_NAME = "adverbs_v2.txt"

class DeterminantMaskProcessor(ProbingMaskProcessor):
  """Processor for the Determinant data set (ASI version)."""
  FILE_NAME = "dets_v2.txt"

class PrepositionMaskTypeProcessor(ProbingMaskProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "time_direction_location_sentences_v2.txt"

class SentLenProcessor(ProbingProcessor):
  """Processor for the Length prediction data set (SentEval version)."""
  FILE_NAME = "sentence_length.txt"

class WCProcessor(ProbingProcessor):
  """Processor for the Word Content analysis data set (SentEval version)."""
  FILE_NAME = "word_content.txt"

class TreeDepthProcessor(ProbingProcessor):
  """Processor for the Tree depth prediction data set (SentEval version)."""
  FILE_NAME = "tree_depth.txt"

class TopConstProcessor(ProbingProcessor):
  """Processor for the Top Constituents prediction data set (SentEval version)."""
  FILE_NAME = "top_constituents.txt"

class BShiftProcessor(ProbingProcessor):
  """Processor for the Word order analysis data set (SentEval version)."""
  FILE_NAME = "bigram_shift.txt"

class TenseProcessor(ProbingProcessor):
  """Processor for the Verb tense prediction data set (SentEval version)."""
  FILE_NAME = "past_present.txt"

class SubjNumProcessor(ProbingProcessor):
  """Processor for the Subject number prediction data set (SentEval version)."""
  FILE_NAME = "subj_number.txt"

class ObjNumProcessor(ProbingProcessor):
  """Processor for the Object number prediction data set (SentEval version)."""
  FILE_NAME = "obj_number.txt"

class SOMOProcessor(ProbingProcessor):
  """Processor for the Semantic odd man out data set (SentEval version)."""
  FILE_NAME = "odd_man_out.txt"

class CoordInvProcessor(ProbingProcessor):
  """Processor for the Coordination Inversion data set (SentEval version)."""
  FILE_NAME = "coordination_inversion.txt"

class AdverbTypeProcessor(ProbingProcessor):
  """Processor for the Adverb Type data set (ASI version)."""
  FILE_NAME = "adverbs.txt"

class ComplexSentenceProcessor(ProbingProcessor):
  """Processor for the Complex Sentence data set (ASI version)."""
  FILE_NAME = "compound_complex_sentences.txt"

class DeterminantProcessor(ProbingProcessor):
  """Processor for the Determinant data set (ASI version)."""
  FILE_NAME = "dets.txt"

class PrepositionTypeProcessor(ProbingProcessor):
  """Processor for the Preposition Type data set (ASI version)."""
  FILE_NAME = "time_direction_location_sentences.txt"
