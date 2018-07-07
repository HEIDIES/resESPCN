import tensorflow as tf
from model import RESESPCN
from reader import Reader
from datetime import datetime
import logging
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input_data_dir', 'data/tfrecords/train.tfrecords',
                       'input data directory')
tf.flags.DEFINE_integer('batch_size', 10,
                        'batch size')
tf.flags.DEFINE_integer('image_size', 256,
                        'image size')
tf.flags.DEFINE_string('load_model', None,
                       'load model directory')
tf.flags.DEFINE_float('learning_rate', 0.001,
                      'learning rate')
tf.flags.DEFINE_integer('num_residual', 16,
                        'num of residual block')


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/" + current_time.lstrip("checkpoints/")
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():
        resespcn = RESESPCN('resESPCN', image_size=FLAGS.image_size,
                            num_residual=FLAGS.num_residual,
                            learning_rate=FLAGS.learning_rate)
        loss = resespcn.loss
        optimizer = resespcn.optimizer
        resespcn.model()

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
        reader = Reader(FLAGS.input_data_dir, batch_size=FLAGS.batch_size)
        images = reader.feed()

        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=config) as sess:
            if FLAGS.load_model is not None:
                checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
                meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
                step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                while not coord.should_stop() and step < 20000:
                    _train = sess.run(images)
                    ls, op, summary = sess.run([loss, optimizer, summary_op],
                                               feed_dict={resespcn.y: _train})
                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    if (step + 1) % 100 == 0:
                        logging.info('-----------Step %d:-------------' % (step + 1))
                        logging.info('      loss     :{}'.format(ls))

                    if (step + 1) % 5000 == 0:
                        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt",
                                               global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                    step += 1

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()

            except Exception as e:
                coord.request_stop(e)

            finally:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
