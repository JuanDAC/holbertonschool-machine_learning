#!/usr/bin/env python3
"""
File Name: 5-train.py - trains a transformer model
"""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """class CustomSchedule"""

    def __init__(self, d_model, warmup_steps=4000):
        """
        Constructor method for CustomSchedule class
        Arguments:
            - d_model: the dimensionality of the model
            - warmup_steps: the number of warmup steps
        """
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        call method for CustomSchedule class
        Arguments:
            - step: the current step
        Returns:
            - the learning rate
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Function that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:
    N the number of blocks in the encoder and decoder
    Returns the trained model 
        - N is the number of blocks in the encoder and decoder
        - dm is the dimensionality of the model
        - h is the number of heads
        - hidden is the number of hidden units in the fully connected layers
        - max_len is the maximum number of tokens per sequence
        - batch_size is the batch size for training
        - epochs is the number of epochs to train for
    """
    dataset = Dataset(batch_size, max_len)
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                         beta_2=0.98, epsilon=1e-9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    transformer = Transformer(N, dm, h, hidden, input_vocab_size,
                              target_vocab_size, max_len, max_len)

    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        encoder_mask, look_ahead_mask, decoder_mask = create_masks(
            inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp, tar_inp, True, encoder_mask,
                                         look_ahead_mask, decoder_mask)
            loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inp, tar)) in enumerate(dataset.data_train):
            train_step(inp, tar)
            if batch % 50 == 0:
                print('Epoch {}, batch {}: loss {} accuracy {}'.format(
                    epoch + 1, batch, train_loss.result(),
                    train_accuracy.result()))
        print('Epoch {}: loss {} accuracy {}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))
    return transformer
