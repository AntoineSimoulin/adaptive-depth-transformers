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
# coding=utf-8
"""Create masked LM/next sentence masked_lm TF examples for ALBERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
from albert import tokenization
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
from collections import namedtuple
import os


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
		"output_file", None,
		"Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
		"output_dir", None,
		"Output directory where dep and pos list vocab are saved.")

flags.DEFINE_string(
		"vocab_file", None,
		"The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string("input_file_mode", "r",
                    "The data format of the input file.")

flags.DEFINE_bool(
		"do_lower_case", True,
		"Whether to lower case the input text. Should be True for uncased "
		"models and False for cased models.")

flags.DEFINE_bool(
		"do_whole_word_mask", True,
		"Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_bool(
		"do_permutation", False,
		"Whether to do the permutation training.")

flags.DEFINE_bool(
		"favor_shorter_ngram", True,
		"Whether to set higher probabilities for sampling shorter ngrams.")

flags.DEFINE_bool(
		"random_next_sentence", False,
		"Whether to use the sentence that's right before the current sentence "
		"as the negative sample for next sentence prection, rather than using "
		"sentences from other random documents.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("ngram", 3, "Maximum number of ngrams to mask.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
		"dupe_factor", 40,
		"Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
		"short_seq_prob", 0.1,
		"Probability of creating sequences which are shorter than the "
		"maximum length.")


class TrainingInstance(object):
		"""A single training instance (sentence pair)."""

		def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
		             is_random_next, token_boundary, upos=[], deps=[], tokens_word_idx=[]):
				self.tokens = tokens
				self.segment_ids = segment_ids
				self.is_random_next = is_random_next
				self.token_boundary = token_boundary
				self.masked_lm_positions = masked_lm_positions
				self.masked_lm_labels = masked_lm_labels
				self.upos = upos
				self.deps = deps
				self.tokens_word_idx = tokens_word_idx

		def __str__(self):
				s = ""
				s += "tokens: %s\n" % (" ".join(
						[tokenization.printable_text(x) for x in self.tokens]))
				s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
				s += "token_boundary: %s\n" % (" ".join(
						[str(x) for x in self.token_boundary]))
				s += "is_random_next: %s\n" % self.is_random_next
				s += "masked_lm_positions: %s\n" % (" ".join(
						[str(x) for x in self.masked_lm_positions]))
				s += "masked_lm_labels: %s\n" % (" ".join(
						[tokenization.printable_text(x) for x in self.masked_lm_labels]))
				if len(self.upos):
						s += "pos_tags: %s\n" % (" ".join(
								[tokenization.printable_text(x) for x in self.upos]))
				if len(self.deps):
						s += "deps: %s\n" % (" ".join(
								[tokenization.printable_text(x) for x in self.deps]))
				if len(self.tokens_word_idx):
						s += "tokens_word_idx: %s\n" % (" ".join(
								[tokenization.printable_text(x) for x in self.tokens_word_idx]))
				s += "\n"
				return s

		def __repr__(self):
				return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files,
                                    vocab_upos, vocab_deps):
		"""Create TF example files from `TrainingInstance`s."""

		upos_2_idx = {upos: idx for (idx, upos) in enumerate(vocab_upos, 1)}
		deps_2_idx = {deps: idx for (idx, deps) in enumerate(vocab_deps, 1)}

		writers = []
		for output_file in output_files:
				writers.append(tf.python_io.TFRecordWriter(output_file))

		writer_index = 0

		total_written = 0
		for (inst_index, instance) in enumerate(instances):
				input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
				input_mask = [1] * len(input_ids)
				segment_ids = list(instance.segment_ids)
				token_boundary = list(instance.token_boundary)
				upos = [upos_2_idx.get(instance.upos[t_idx], 0) for t_idx in instance.tokens_word_idx]
				deps = [deps_2_idx.get(instance.deps[t_idx], 0) for t_idx in instance.tokens_word_idx]

				assert len(input_ids) <= max_seq_length

				while len(input_ids) < max_seq_length:
						input_ids.append(0)
						input_mask.append(0)
						segment_ids.append(0)
						token_boundary.append(0)
						upos.append(0)
						deps.append(0)

				assert len(input_ids) == max_seq_length
				assert len(input_mask) == max_seq_length
				assert len(segment_ids) == max_seq_length
				assert len(upos) == max_seq_length
				assert len(deps) == max_seq_length

				masked_lm_positions = list(instance.masked_lm_positions)
				masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
				masked_lm_weights = [1.0] * len(masked_lm_ids)

				multiplier = 1 + int(FLAGS.do_permutation)
				while len(masked_lm_positions) < max_predictions_per_seq * multiplier:
						masked_lm_positions.append(0)
						masked_lm_ids.append(0)
						masked_lm_weights.append(0.0)

				sentence_order_label = int(instance.is_random_next)  # 1 if instance.is_random_next else 0

				features = collections.OrderedDict()
				features["input_ids"] = create_int_feature(input_ids)
				features["input_mask"] = create_int_feature(input_mask)
				features["segment_ids"] = create_int_feature(segment_ids)
				features["token_boundary"] = create_int_feature(token_boundary)
				features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
				features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
				features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
				# Note: We keep this feature name `next_sentence_labels` to be compatible
				# with the original data created by lanzhzh@. However, in the ALBERT case
				# it does contain sentence_order_label.
				features["next_sentence_labels"] = create_int_feature(
						[sentence_order_label])

				features["upos_ids"] = create_int_feature(upos)
				features["deps_ids"] = create_int_feature(deps)

				tf_example = tf.train.Example(features=tf.train.Features(feature=features))

				writers[writer_index].write(tf_example.SerializeToString())
				writer_index = (writer_index + 1) % len(writers)

				total_written += 1

				if inst_index < 20:
						tf.logging.info("*** Example ***")
						tf.logging.info("tokens: %s" % " ".join(
								[tokenization.printable_text(x) for x in instance.tokens]))

						for feature_name in features.keys():
								feature = features[feature_name]
								values = []
								if feature.int64_list.value:
										values = feature.int64_list.value
								elif feature.float_list.value:
										values = feature.float_list.value
								tf.logging.info(
										"%s: %s" % (feature_name, " ".join([str(x) for x in values])))

		for writer in writers:
				writer.close()

		tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
		feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
		return feature


def create_float_feature(values):
		feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
		return feature


conll_header = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
Conll = namedtuple('Conll', conll_header)


def _read_conll(input_file):
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


def encode_conll_keys(dataset, key):
		conll_keys = {}
		for sentence in dataset:
				for token in sentence:
						token_key = getattr(token, key)
						conll_keys[token_key] = conll_keys.get(token_key, 0) + 1
		return [x[0] for x in sorted(conll_keys.items(), key=lambda kv: kv[1], reverse=True)]


def save_conll_encoder(output_dir, encoder, key):
		with open(os.path.join(output_dir, '{}.conll.encoder'.format(key)), 'w') as f:
				_ = [f.write('{}\n'.format(k)) for k in encoder]
		return


def load_conll_encoder(output_dir, encoder, key):
		with open(os.path.join(output_dir, '{}.conll.encoder'.format(key)), 'r') as f:
				conll_keys = f.readlines()
		conll_keys = [k.rstrip('\n') for k in conll_keys]
		return conll_keys


def create_training_instances(input_files, tokenizer, max_seq_length, output_dir):
		"""Create `TrainingInstance`s from raw text."""
		all_documents = []
		all_upos = []
		all_deps = []

		for input_file in input_files:
				lines = _read_conll(input_file)
				for line in lines:
						if not FLAGS.spm_model_file:
								line_ = tokenization.convert_to_unicode(' '.join([token.form for token in line]))
						if FLAGS.spm_model_file:
								line_ = tokenization.preprocess_text(' '.join([token.form for token in line]), lower=FLAGS.do_lower_case)
						tokens = tokenizer.tokenize(line_)
						if tokens:
								all_documents.append(tokens)
								all_upos.append([token.upos for token in line])
								all_deps.append([token.deprel for token in line])
		all_documents = [x for x in all_documents if x]
		all_upos = [x for x in all_upos if x]
		all_deps = [x for x in all_deps if x]

		vocab_upos = encode_conll_keys(lines, 'upos')
		_ = save_conll_encoder(output_dir, vocab_upos, 'upos')

		vocab_deps = encode_conll_keys(lines, 'deprel')
		_ = save_conll_encoder(output_dir, vocab_deps, 'deprel')

		instances = create_instances_from_documents(
				all_documents, all_upos, all_deps, max_seq_length)

		return instances, vocab_deps, vocab_upos


def create_instances_from_documents(
								all_documents, all_upos, all_deps, max_seq_length):
		"""Creates `TrainingInstance`s for a single document."""

		instances = []
		i = 0
		while i < len(all_documents):

				# is_random_next = False
				tokens_a = all_documents[i]
				upos_a = all_upos[i]
				deps_a = all_deps[i]

				if len(tokens_a) > max_seq_length - 2:
						# Account for [CLS], [SEP]
						tokens_a = tokens_a[0:(max_seq_length - 2)]
						upos_a = upos_a[0:(max_seq_length - 2)]
						deps_a = deps_a[0:(max_seq_length - 2)]

				assert len(tokens_a) >= 1

				tokens = []
				segment_ids = []
				upos = []
				deps = []
				tokens_is_word = []

				tokens.append("[CLS]")
				upos.append("[CLS]")
				deps.append("[CLS]")
				segment_ids.append(0)
				for token_idx, (token, upo, dep) in enumerate(zip(tokens_a, upos_a, deps_a)):
						tokens.append(token)
						upos.append(upo)
						deps.append(dep)
						segment_ids.append(0)
						if token_idx == 0:
								tokens_is_word.append(1)
						else:
								tokens_is_word.append(int(is_start_piece(token)))
				tokens.append("[SEP]")
				upos.append("[CLS]")
				deps.append("[CLS]")
				segment_ids.append(0)
				# tokens_is_word = [1 if _is_start_piece_sp(piece) else 0 for piece in tokens]
				tokens_word_idx = list(np.cumsum(tokens_is_word))
				tokens_word_idx.insert(0, 0)
				tokens_word_idx.append(0)

				instance = TrainingInstance(
						tokens=tokens,
						segment_ids=segment_ids,
						upos=upos,
						deps=deps,
						tokens_word_idx=tokens_word_idx,
						is_random_next=-1,
						token_boundary=[],
						masked_lm_positions=[],
						masked_lm_labels=[])
				instances.append(instance)

				i += 1

		return instances


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


def is_start_piece(piece):
		if FLAGS.spm_model_file:
				return _is_start_piece_sp(piece)
		else:
				return _is_start_piece_bert(piece)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
		"""Truncates a pair of sequences to a maximum sequence length."""
		while True:
				total_length = len(tokens_a) + len(tokens_b)
				if total_length <= max_num_tokens:
						break

				trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
				assert len(trunc_tokens) >= 1

				# We want to sometimes truncate from the front and sometimes from the
				# back to add more randomness and avoid biases.
				if rng.random() < 0.5:
						del trunc_tokens[0]
				else:
						trunc_tokens.pop()


def main(_):
		tf.logging.set_verbosity(tf.logging.INFO)

		tokenizer = tokenization.FullTokenizer(
				vocab_file=FLAGS.vocab_file,
				do_lower_case=FLAGS.do_lower_case,
				spm_model_file=FLAGS.spm_model_file)

		input_files = []
		for input_pattern in FLAGS.input_file.split(","):
				input_files.extend(tf.gfile.Glob(input_pattern))

		tf.logging.info("*** Reading from input files ***")
		for input_file in input_files:
				tf.logging.info("  %s", input_file)

		instances, vocab_deps, vocab_upos = create_training_instances(input_files, tokenizer,
		                                                              FLAGS.max_seq_length, FLAGS.output_dir)
		tf.logging.info("number of instances: %i", len(instances))

		output_files = FLAGS.output_file.split(",")
		tf.logging.info("*** Writing to output files ***")
		for output_file in output_files:
				tf.logging.info("  %s", output_file)

		write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
		                                FLAGS.max_predictions_per_seq, output_files,
		                                vocab_upos, vocab_deps)


if __name__ == "__main__":
		flags.mark_flag_as_required("input_file")
		flags.mark_flag_as_required("output_file")
		flags.mark_flag_as_required("vocab_file")
		flags.mark_flag_as_required("output_dir")
		tf.app.run()
