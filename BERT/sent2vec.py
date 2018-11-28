from . import extract_features as helper
import numpy as np
from . import modeling
from . import tokenization
import tensorflow as tf
layer_indexes = [-1]


flags = tf.flags

FLAGS = flags.FLAGS

BERTPAT = "/Users/yuxiangli/lib/bert/uncased_L-12_H-768_A-12/"
flags.DEFINE_string(
    "bert_config_file", BERTPAT+"bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", BERTPAT+"bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", BERTPAT+"vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class VecGenerator():
    def __init__(self):

        bert_config = modeling.BertConfig.from_json_file(
            FLAGS.bert_config_file)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        run_config = tf.contrib.tpu.RunConfig(
            master=FLAGS.master,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=FLAGS.num_tpu_cores,
                per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

        model_fn = helper.model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            layer_indexes=layer_indexes,  # 我们只要最后的语义表示
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            predict_batch_size=FLAGS.batch_size)

    def __call__(self, sent):
        # line = tokenization.convert_to_unicode(sent)
        example = helper.InputExample(unique_id=0, text_a=sent, text_b="")

        features = helper.convert_examples_to_features(
            examples=[example], seq_length=FLAGS.max_seq_length, tokenizer=self.tokenizer)

        input_fn = helper.input_fn_builder(
            features=features, seq_length=FLAGS.max_seq_length)

        result = list(self.estimator.predict(
            input_fn, yield_single_examples=True))[0]

        layer_output = result["layer_output_0"]
        # layer_outputT = layer_output[1:len(features[0].tokens)-1, :]
        # sentence_rep = np.sum(layer_outputT, axis=0)
        sentence_rep=layer_output[0]
        return sentence_rep
