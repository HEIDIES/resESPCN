import tensorflow as tf
import utils
import matplotlib.pyplot as plt


class Reader:
    def __init__(self, tfrecords_file, image_size=256, min_queue_examples=100, batch_size=4,
                 num_threads=12, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.name = name
        self.reader = tf.TFRecordReader()

    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])

            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/train': tf.FixedLenFeature([], tf.string),
                })

            train_buffer = features['image/train']
            train = tf.image.decode_jpeg(train_buffer, channels=3)
            train = self._preprocess(train)
            train = tf.train.shuffle_batch(
                [train], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )

            # tf.summary.image('_input', train)
        return train

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = utils.convert2float(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return image


def test_reader():
    train_file_1 = 'data/tfrecords/train.tfrecords'

    with tf.Graph().as_default():
        reader1 = Reader(train_file_1, batch_size=1)
        image_train = reader1.feed()
        image_train = tf.squeeze(image_train, 0)
        image = utils.convert2int(image_train)
        # image_label = tf.squeeze(image_label, 0)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop() and step < 1:
                train = sess.run(image)
                print(train.shape)
                f, a = plt.subplots(1, 1)
                # for i in range(1):
                a.imshow(train)
                plt.show()
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
