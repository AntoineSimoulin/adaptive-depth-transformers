from collections import namedtuple
import os
from albert import tokenization, modeling
from albert.classifier_utils import PaddingInputExample
import tensorflow as tf
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import metrics as contrib_metrics

from albert import optimization
import collections
import six
import numpy as np


conll_header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
Conll = namedtuple('Conll', conll_header)

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, deps_a, upos_a,
               text_b=None, deps_b=None, upos_b=None, label=None):
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
    self.deps_a = deps_a
    self.upos_a = upos_a
    self.text_b = text_b
    self.deps_b = deps_b
    self.upos_b = upos_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def __init__(self, use_spm, do_lower_case):
    super(DataProcessor, self).__init__()
    self.use_spm = use_spm
    self.do_lower_case = do_lower_case

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
    raise NotImplementedError()

  @classmethod
  def _read_conll(cls, input_file, quotechar=None):
    """Reads a conll formated file."""
    lines = []
    with open(input_file, 'r') as f:
      line = f.readline()
      sentence = []
      while line:
        if line.rstrip('\n'):
          sentence.append(Conll(*line.rstrip('\n').split('\t')))
        else:
          lines.append(sentence)
          sentence = []
        line = f.readline()
    return lines

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def process_text(self, text):
    if self.use_spm:
      return tokenization.preprocess_text(text, lower=self.do_lower_case)
    else:
      return tokenization.convert_to_unicode(text)


class ConllProcessor(DataProcessor):
  """Processor for the Penn Tree Bank data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_conll(os.path.join(data_dir, "train.English.pred.conll")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_conll(os.path.join(data_dir, "dev.English.pred.conll")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_conll(os.path.join(data_dir, "test.English.pred.conll")),
        "test")

  def get_all_examples(self, data_dir):
    train_examples = self.get_train_examples(data_dir)
    dev_examples = self.get_dev_examples(data_dir)
    test_examples = self.get_test_examples(data_dir)
    return train_examples + dev_examples + test_examples

  def get_labels(self):
    """See base class."""
    return [0]  # random label

  def _create_examples(self, sentences, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, sentence) in enumerate(sentences):
      guid = self.process_text(str(i))
      text_a = self.process_text(' '.join([token.form for token in sentence]))
      deps_a = [token.deprel for token in sentence]
      upos_a = [token.upos for token in sentence]
      text_b = None
      deps_b = None
      upos_b = None
      label = 0
      examples.append(
          InputExample(guid=guid,
                       text_a=text_a, deps_a=deps_a, upos_a=upos_a,
                       text_b=text_b, deps_b=deps_b, upos_b=upos_b,
                       label=label))  # , tags_a=tags_a, tags_b=tags_b
    return examples


########################################################################################################################

def convert_single_example_with_mask(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_name):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
      input_ids=[0] * max_seq_length,
      input_mask=[0] * max_seq_length,
      segment_ids=[0] * max_seq_length,
      label_id=0,
      n_intervening=[0],
      distance=[0],
      is_real_example=False)

  if task_name != "sts-b":
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in ALBERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.

  tokens_is_word = []
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token_idx, token in enumerate(tokens_a):
    tokens.append(token)
    segment_ids.append(0)
    if token_idx == 0:
      tokens_is_word.append(1)
    else:
      tokens_is_word.append(int(is_start_piece(token)))
  tokens.append("[SEP]")
  segment_ids.append(0)
  tokens_is_word.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
      tokens_is_word.append(int(is_start_piece(token)))
    tokens.append("[SEP]")
    segment_ids.append(1)
    tokens_word_idx.append(0)

  tokens_word_idx = list(np.cumsum(tokens_is_word))
  tokens_word_idx.insert(0, 0)


  if ex_index < 5:
    tf.logging.info("tokens_word_idx: %s"% " ".join([str(x) for x in tokens_word_idx]))

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  if hasattr(example, 'words_to_mask_idx'):
    if example.words_to_mask_idx is not None:
      for idx, tok_idx in enumerate(tokens_word_idx):
        if tok_idx in example.words_to_mask_idx:
          input_ids[idx] = tokenizer.vocab['[MASK]']

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  if task_name != "sts-b":
    label_id = label_map[example.label]
  else:
    label_id = example.label

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("words_to_mask_idx: %s" % " ".join([str(x) for x in example.words_to_mask_idx]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    if hasattr(example, 'n_intervening') and hasattr(example, 'distance'):
      tf.logging.info("distance: %s" % (example.distance))
      tf.logging.info("n_intervening: %s" % (example.n_intervening))


  if hasattr(example, 'n_intervening') and hasattr(example, 'distance'):
    feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      n_intervening=example.n_intervening,
      distance=example.distance,
      is_real_example=True)
  else:
    feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def _is_start_piece_sp(piece):
  """Check if the current word piece is the starting piece (sentence piece)."""
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  special_pieces.add(u"€".encode("utf-8"))
  special_pieces.add(u"£".encode("utf-8"))
  # Note(mingdachen):
  # For foreign characters, we always treat them as a whole piece.
  english_chars = set(list("abcdefghijklmnopqrstuvwxyz"))
  if (six.ensure_str(piece).startswith("▁") or
          six.ensure_str(piece).startswith("<") or piece in special_pieces or
          not all([i.lower() in english_chars.union(special_pieces)
                   for i in piece])):
    return True
  else:
    return False


def _is_start_piece_bert(piece):
  """Check if the current word piece is the starting piece (BERT)."""
  # When a word has been split into
  # WordPieces, the first token does not have any marker and any subsequence
  # tokens are prefixed with ##. So whenever we see the ## token, we
  # append it to the previous set of word indexes.
  return not six.ensure_str(piece).startswith("##")


def is_start_piece(piece, spm_model_file=True):
  if spm_model_file:
    return _is_start_piece_sp(piece)
  else:
    return _is_start_piece_bert(piece)


def file_based_convert_examples_to_features_with_mask(
        examples, label_list, max_seq_length, tokenizer, output_file, task_name):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example_with_mask(ex_index, example, label_list,
                                               max_seq_length, tokenizer, task_name)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_float_feature([feature.label_id]) \
      if task_name == "sts-b" else create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
      [int(feature.is_real_example)])
    if feature.n_intervening is not None:
      features["n_intervening"] = create_int_feature(feature.n_intervening)
    if feature.distance is not None:
      features["distance"] = create_int_feature(feature.distance)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


########################################################################################################################


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, task_name, vocab_deps, vocab_upos):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  vocab_upos_list = load_conll_encoder(vocab_upos)
  vocab_deps_list = load_conll_encoder(vocab_deps)
  upos_2_idx = {upos: idx for (idx, upos) in enumerate(vocab_upos_list, 1)}
  deps_2_idx = {deps: idx for (idx, deps) in enumerate(vocab_deps_list, 1)}

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer, task_name,
                                     deps_2_idx, upos_2_idx)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["deps_ids"] = create_int_feature(feature.deps_ids)
    features["upos_ids"] = create_int_feature(feature.upos_ids)
    features["label_ids"] = create_float_feature([feature.label_id])\
        if task_name == "sts-b" else create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               deps_ids=None,
               upos_ids=None,
               guid=None,
               example_id=None,
               is_real_example=True,
               n_intervening=None,
               distance=None,
               verb_pos=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.deps_ids = deps_ids
    self.upos_ids = upos_ids
    self.label_id = label_id
    self.example_id = example_id
    self.guid = guid
    self.is_real_example = is_real_example
    self.n_intervening = n_intervening
    self.distance = distance
    self.verb_pos = verb_pos


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer, task_name, deps_2_idx, upos_2_idx):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
		    deps_ids=[0] * max_seq_length,
        upos_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  if task_name != "sts-b":
    label_map = {}
    for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  deps_a = example.deps_a
  upos_a = example.upos_a
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)
    deps_b = example.deps_b
    upos_b = example.upos_b

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, deps_a, upos_a, tokens_b, deps_b, upos_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
      deps_a = deps_a[0:(max_seq_length - 2)]
      upos_a = upos_a[0:(max_seq_length - 2)]



  # The convention in ALBERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  deps_ids = []
  upos_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  deps_ids.append("[CLS]")
  upos_ids.append("[CLS]")
  for token, dep, upo in zip(tokens_a, deps_a, upos_a):
    tokens.append(token)
    deps_ids.append(dep)
    upos_ids.append(upo)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)
  deps_ids.append(0)
  upos_ids.append(0)

  if tokens_b:
    for token, dep, upo in zip(tokens_b, deps_b, upos_b):
      tokens.append(token)
      deps_ids.append(dep)
      upos_ids.append(upo)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    deps_ids.append("[CLS]")
    upos_ids.append("[CLS]")

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  deps_ids = [deps_2_idx.get(dep, 0) for dep in deps_ids]
  upos_ids = [upos_2_idx.get(upo, 0) for upo in upos_ids]

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    deps_ids.append(0)
    upos_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(deps_ids) == max_seq_length
  assert len(upos_ids) == max_seq_length

  if task_name != "sts-b":
    label_id = label_map[example.label]
  else:
    label_id = example.label

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("deps_ids: %s" % " ".join([str(x) for x in deps_ids]))
    tf.logging.info("upos_ids: %s" % " ".join([str(x) for x in upos_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
		  deps_ids=deps_ids,
      upos_ids=upos_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def _truncate_seq_pair(tokens_a, deps_a, upos_a, tokens_b, deps_b, upos_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
      deps_a.pop()
      upos_a.pop()
    else:
      tokens_b.pop()
      deps_b.pop()
      upos_b.pop()


def file_based_input_fn_builder_with_tags(input_file, seq_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1, add_dep_and_pos=True):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.float32 if task_name == "sts-b" else tf.int64

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      # "n_intervening": tf.FixedLenFeature([1], tf.int64),
      # "distance": tf.FixedLenFeature([1], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "label_ids": tf.FixedLenFeature([], labeltype),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  if add_dep_and_pos:
    name_to_features.update(
      {"upos_ids": tf.FixedLenFeature([seq_length], tf.int64, default_value=[0] * seq_length)})
    name_to_features.update(
      {"deps_ids": tf.FixedLenFeature([seq_length], tf.int64, default_value=[0] * seq_length)})

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn



def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, task_name, use_tpu, bsz,
                                multiple=1,):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  labeltype = tf.float32 if task_name == "sts-b" else tf.int64

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length * multiple], tf.int64),
      "label_ids": tf.FixedLenFeature([], labeltype),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if use_tpu:
      batch_size = params["batch_size"]
    else:
      batch_size = bsz

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        contrib_data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _create_model_from_scratch(albert_config, is_training, input_ids,
                               input_mask, segment_ids, use_one_hot_embeddings,
                               use_einsum):
  """Creates an ALBERT model from scratch/config."""
  model = modeling.AlbertModelWithACT(
      config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_einsum=use_einsum)
  return (model.get_pooled_output(), model.get_sequence_output(), model.get_n_updates(), model.get_time_penaly())


def create_albert(albert_config, is_training, input_ids, input_mask,
                  segment_ids, use_one_hot_embeddings, use_einsum):
  """Creates an ALBERT, either from scratch."""

  tf.logging.info("creating model from albert_config")
  return _create_model_from_scratch(albert_config, is_training, input_ids,
                                    input_mask, segment_ids,
                                    use_one_hot_embeddings, use_einsum)


# def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
#                  labels, num_labels, use_one_hot_embeddings, task_name,
#                  hub_module):
#   """Creates a classification model."""
#   (output_layer, _, n_updates) = create_albert(
#       albert_config=albert_config,
#       is_training=is_training,
#       input_ids=input_ids,
#       input_mask=input_mask,
#       segment_ids=segment_ids,
#       use_one_hot_embeddings=use_one_hot_embeddings,
#       use_einsum=True)
#
#   return (output_layer, n_updates)


def load_conll_encoder(vocab_path):
  with open(os.path.join(vocab_path), 'r') as f:
      conll_keys = f.readlines()
  conll_keys = [k.rstrip('\n') for k in conll_keys]
  return conll_keys


def create_model(albert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, task_name, tau,
                 hub_module, use_bos=True):
  """Creates a classification model."""
  (output_layer, output_sequence, n_updates, time_penalty) = create_albert(
      albert_config=albert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings,
      use_einsum=True)
  # `sequence_output` shape = [batch_size, seq_length, hidden_size].

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      if use_bos:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
      else:
        output_sequence = tf.nn.dropout(output_sequence, keep_prob=0.9)

    if use_bos:
      logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    else:
      # zeros padding index
      # output_sequence = tf.where(
      #   tf.cast(tf.tile(tf.expand_dims(input_mask, axis=2), tf.constant([1, 1, hidden_size])), tf.bool),
      #   output_sequence,
      #   tf.zeros_like(output_sequence, dtype=tf.float32))  #  * tf.reduce_min(output_sequence)

      output_sequence = tf.math.reduce_mean(output_sequence, axis=1)
      # output_sequence = tf.layers.dense(
      #   output_sequence,
      #   albert_config.hidden_size,
      #   activation=tf.tanh,
      #   kernel_initializer=modeling.create_initializer(albert_config.initializer_range))

      logits = tf.matmul(output_sequence, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    if task_name != "sts-b":
      probabilities = tf.nn.softmax(logits, axis=-1)
      predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
      log_probs = tf.nn.log_softmax(logits, axis=-1)
      one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    else:
      probabilities = logits
      logits = tf.squeeze(logits, [-1])
      predictions = logits
      per_example_loss = tf.square(logits - labels)

    task_loss = tf.reduce_mean(per_example_loss)
    time_penalty_loss = tf.reduce_mean(time_penalty * tau)
    total_loss = time_penalty_loss + task_loss

    return (total_loss, per_example_loss, probabilities, logits, predictions, n_updates)


def model_fn_builder(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name, hub_module=None,
                     optimizer="adamw", tau=5e-4, use_bos=True):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, probabilities, logits, predictions, n_updates) = \
        create_model(albert_config, is_training, input_ids, input_mask,
                     segment_ids, label_ids, num_labels, use_one_hot_embeddings,
                     task_name, tau, hub_module, use_bos)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn_tags(tags_2_idx, n_updates, tags_ids):
        tags_metrics = {}
        for (k, v) in tags_2_idx.items():
          tags_idx = tf.where(tf.equal(tags_ids, v))
          tags_metrics[k] = tf.metrics.mean(tf.gather_nd(n_updates, tags_idx))
        return tags_metrics

      if task_name not in ["sts-b", "cola"]:
        def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                      n_updates):
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          loss = tf.metrics.mean(
              values=per_example_loss, weights=is_real_example)

          metrics = {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
          }

          return metrics

      elif task_name == "sts-b":
        def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                      n_updates):
          """Compute Pearson correlations for STS-B."""
          # Display labels and predictions
          concat1 = contrib_metrics.streaming_concat(logits)
          concat2 = contrib_metrics.streaming_concat(label_ids)

          # Compute Pearson correlation
          pearson = contrib_metrics.streaming_pearson_correlation(
              logits, label_ids, weights=is_real_example)

          # Compute MSE
          # mse = tf.metrics.mean(per_example_loss)
          mse = tf.metrics.mean_squared_error(
              label_ids, logits, weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          metrics = {
            "pred": concat1,
            "label_ids": concat2,
            "pearson": pearson,
            "MSE": mse,
            "eval_loss": loss}

          return metrics

      elif task_name == "cola":

        def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                      n_updates):
          """Compute Matthew's correlations for COLA."""
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
          tp, tp_op = tf.metrics.true_positives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          tn, tn_op = tf.metrics.true_negatives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          fp, fp_op = tf.metrics.false_positives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          fn, fn_op = tf.metrics.false_negatives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)

          # Compute Matthew's correlation
          mcc = tf.div_no_nan(
              tp * tn - fp * fn,
              tf.pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5))

          # Compute accuracy
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          metrics = {
            "matthew_corr": (mcc, tf.group(tp_op, tn_op, fp_op, fn_op)),
            "eval_accuracy": accuracy,
            "eval_loss": loss,}

          return metrics

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example,
                       n_updates])
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
            "labels": label_ids,
            "probabilities": probabilities,
            "predictions": predictions,
            # "output_layer": output_layer,
            'input_ids': input_ids
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def model_fn_builder_with_tags(albert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, task_name, hub_module=None,
                     optimizer="adamw", vocab_upos=[], vocab_deps=[], tau=5e-4, use_bos=True):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    if ("upos_ids" in features) and ("deps_ids" in features):
      upos_ids = features["upos_ids"]
      deps_ids = features["deps_ids"]
      add_dep_and_pos = True
    else:
      add_dep_and_pos = False
      upos_ids = tf.zeros_like(input_ids, dtype=tf.int32)
      deps_ids = tf.zeros_like(input_ids, dtype=tf.int32)

    # if ("n_intervening" in features) and ("distance" in features):
    #   n_intervening = features["n_intervening"]
    #   distance = features["distance"]
    # else:
    #   n_intervening = tf.zeros([1], dtype=tf.int32)
    #   distance = tf.zeros([1], dtype=tf.int32)

    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, probabilities, logits, predictions, n_updates) = \
        create_model(albert_config, is_training, input_ids, input_mask,
                     segment_ids, label_ids, num_labels, use_one_hot_embeddings,
                     task_name, tau, hub_module, use_bos)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      vocab_upos_list = load_conll_encoder(vocab_upos)
      vocab_deps_list = load_conll_encoder(vocab_deps)
      upos_2_idx = {upos: idx for (idx, upos) in enumerate(vocab_upos_list, 1)}
      deps_2_idx = {deps: idx for (idx, deps) in enumerate(vocab_deps_list, 1)}

      def metric_fn_tags(tags_2_idx, n_updates, tags_ids):
        tags_metrics = {}
        for (k, v) in tags_2_idx.items():
          tags_idx = tf.where(tf.equal(tags_ids, v))
          tags_metrics[k] = tf.metrics.mean(tf.gather_nd(n_updates, tags_idx))
        return tags_metrics

      if task_name not in ["sts-b", "cola"]:
        def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                      n_updates, deps_ids, upos_ids):
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          loss = tf.metrics.mean(
              values=per_example_loss, weights=is_real_example)

          metrics = {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
          }

          if add_dep_and_pos:
            deps_metrics = metric_fn_tags(deps_2_idx, n_updates, deps_ids)
            for (k, v) in deps_2_idx.items():
              metrics.update({"n_updates_deps/{}".format(k): deps_metrics[k]})

            upos_metrics = metric_fn_tags(upos_2_idx, n_updates, upos_ids)
            for (k, v) in upos_2_idx.items():
              metrics.update({"n_updates_upos/{}".format(k): upos_metrics[k]})

          return metrics

      elif task_name == "sts-b":
        def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                      n_updates, deps_ids, upos_ids):
          """Compute Pearson correlations for STS-B."""
          # Display labels and predictions
          concat1 = contrib_metrics.streaming_concat(logits)
          concat2 = contrib_metrics.streaming_concat(label_ids)

          # Compute Pearson correlation
          pearson = contrib_metrics.streaming_pearson_correlation(
              logits, label_ids, weights=is_real_example)

          # Compute MSE
          # mse = tf.metrics.mean(per_example_loss)
          mse = tf.metrics.mean_squared_error(
              label_ids, logits, weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          metrics = {
            "pred": concat1,
            "label_ids": concat2,
            "pearson": pearson,
            "MSE": mse,
            "eval_loss": loss}

          if add_dep_and_pos:
            deps_metrics = metric_fn_tags(deps_2_idx, n_updates, deps_ids)
            for (k, v) in deps_2_idx.items():
              metrics.update({"n_updates_deps/{}".format(k): deps_metrics[k]})

            upos_metrics = metric_fn_tags(upos_2_idx, n_updates, upos_ids)
            for (k, v) in upos_2_idx.items():
              metrics.update({"n_updates_upos/{}".format(k): upos_metrics[k]})

          return metrics

      elif task_name == "cola":

        def metric_fn(per_example_loss, label_ids, logits, is_real_example,
                      n_updates, deps_ids, upos_ids):
          """Compute Matthew's correlations for COLA."""
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
          # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
          tp, tp_op = tf.metrics.true_positives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          tn, tn_op = tf.metrics.true_negatives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          fp, fp_op = tf.metrics.false_positives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)
          fn, fn_op = tf.metrics.false_negatives(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)

          # Compute Matthew's correlation
          mcc = tf.div_no_nan(
              tp * tn - fp * fn,
              tf.pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5))

          # Compute accuracy
          accuracy = tf.metrics.accuracy(
              labels=label_ids, predictions=predictions,
              weights=is_real_example)

          loss = tf.metrics.mean(
              values=per_example_loss,
              weights=is_real_example)

          metrics = {
            "matthew_corr": (mcc, tf.group(tp_op, tn_op, fp_op, fn_op)),
            "eval_accuracy": accuracy,
            "eval_loss": loss,}

          if add_dep_and_pos:
            deps_metrics = metric_fn_tags(deps_2_idx, n_updates, deps_ids)
            for (k, v) in deps_2_idx.items():
              metrics.update({"n_updates_deps/{}".format(k): deps_metrics[k]})

            upos_metrics = metric_fn_tags(upos_2_idx, n_updates, upos_ids)
            for (k, v) in upos_2_idx.items():
              metrics.update({"n_updates_upos/{}".format(k): upos_metrics[k]})

          return metrics

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example,
                       n_updates, deps_ids, upos_ids])
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
            "labels": label_ids,
            "probabilities": probabilities,
            "predictions": predictions,
            # "output_layer": output_layer,
            # "n_intervening": n_intervening,
            # "distance": distance,
            "n_updates": n_updates,
            "upos_ids": upos_ids,
            "deps_ids": deps_ids,
            'input_ids': input_ids
          },
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn