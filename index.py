import numpy as np
import tensorflow as tf


def argmax_accuracy(y_true, y_pred):
    labels = tf.argmax(y_pred, axis=-1)
    true_probs = y_pred.numpy()[np.arange(y_true.shape[0])[
        :, None], np.arange(y_true.shape[1]), y_true]
    return tf.reduce_mean(tf.where(labels == y_true, tf.ones_like(labels, dtype=tf.float32), tf.zeros_like(labels, dtype=tf.float32)))


def sampled_accuracy(y_true, y_pred):
    labels = tf.argmax(y_pred, axis=-1)
    true_probs = y_pred.numpy()[np.arange(y_true.shape[0])[
        :, None], np.arange(y_true.shape[1]), y_true]
    return tf.reduce_mean(true_probs)


def normalize(tensor):
    return tensor / (tf.reduce_sum(tensor, axis=-1, keepdims=True) + 1e-7)


def increment_program_counter(tensor, shift=1, axis=1):
    return tf.roll(tensor, shift=shift, axis=axis)


def p_brp(accumulator):
    n = accumulator.shape[-1]  # Get the length of the last dimension

    # Create a boolean mask for the condition (last index between 0 and n//2)
    mask = tf.range(n) < (n // 2)

    # Expand the mask to match the shape of the tensor
    # Shape (1, n) to match the tensor shape
    mask = mask[tf.newaxis, tf.newaxis, :]
    # Repeat the mask for each row
    mask = tf.tile(mask, [accumulator.shape[0], accumulator.shape[1], 1])

    # Zero out elements where the mask is True
    return tf.where(mask, accumulator, tf.zeros_like(accumulator))


def p_brz(accumulator):
    n = accumulator.shape[-1]  # Get the length of the last dimension

    # Create a boolean mask for the condition (last index between 0 and n//2)
#     mask = tf.cast(tf.range(n) == 0, tf.float32)
    mask = tf.range(n) == 0

    # Expand the mask to match the shape of the tensor
    # Shape (1, n) to match the tensor shape
    mask = tf.reshape(mask, [1, 1, -1])
    # Repeat the mask for each row
    mask = tf.tile(mask, [accumulator.shape[0], accumulator.shape[1], 1])

    # Zero out elements where the mask is True
    return tf.where(mask, accumulator, tf.zeros_like(accumulator))


def add(memory, accumulator):
    N = tf.einsum('bij,bjk->bik', memory, memory)
    A = accumulator
    n = memory.shape[-1]
    b = memory.shape[0]

    probabilities = N[:, :, tf.newaxis, :]*A[:, :, :, tf.newaxis]
    flattened_probabilities = tf.reshape(probabilities, [b, n, n*n])
    indices = tf.range(n)
    combined_indices = (indices[:, tf.newaxis]+indices[tf.newaxis, :]) % n
    combined_indices = combined_indices[tf.newaxis, tf.newaxis, :, :]
    combined_indices = tf.tile(combined_indices, [b, n, 1, 1])
    flattened_indices = tf.reshape(combined_indices, [b, n, n*n])

    batch_indices = tf.range(b)[:, tf.newaxis, tf.newaxis]
    batch_indices = tf.tile(batch_indices, [1, n, n*n])
    position_indices = tf.range(n)[tf.newaxis, :, tf.newaxis]
    position_indices = tf.tile(position_indices, [b, 1, n*n])

    final_indices = tf.stack(
        [batch_indices, position_indices, flattened_indices], axis=-1)
    output_shape = [b, n, n]

    return tf.scatter_nd(final_indices, flattened_probabilities, shape=output_shape)


def sub(memory, accumulator):
    N = tf.einsum('bij,bjk->bik', memory, memory)
    A = accumulator
    n = memory.shape[-1]
    b = memory.shape[0]

    probabilities = N[:, :, tf.newaxis, :]*A[:, :, :, tf.newaxis]
    flattened_probabilities = tf.reshape(probabilities, [b, n, n*n])
    indices = tf.range(n)
    combined_indices = (indices[:, tf.newaxis]-indices[tf.newaxis, :]) % n
    combined_indices = combined_indices[tf.newaxis, tf.newaxis, :, :]
    combined_indices = tf.tile(combined_indices, [b, n, 1, 1])
    flattened_indices = tf.reshape(combined_indices, [b, n, n*n])

    batch_indices = tf.range(b)[:, tf.newaxis, tf.newaxis]
    batch_indices = tf.tile(batch_indices, [1, n, n*n])
    position_indices = tf.range(n)[tf.newaxis, :, tf.newaxis]
    position_indices = tf.tile(position_indices, [b, 1, n*n])

    final_indices = tf.stack(
        [batch_indices, position_indices, flattened_indices], axis=-1)
    output_shape = [b, n, n]

    return tf.scatter_nd(final_indices, flattened_probabilities, shape=output_shape)


def mul(memory, accumulator):
    N = tf.einsum('bij,bjk->bik', memory, memory)
    A = accumulator
    n = memory.shape[-1]
    b = memory.shape[0]

    probabilities = N[:, :, tf.newaxis, :]*A[:, :, :, tf.newaxis]
    flattened_probabilities = tf.reshape(probabilities, [b, n, n*n])
    indices = tf.range(n)
    combined_indices = indices[:, tf.newaxis]*indices[tf.newaxis, :] % n
    combined_indices = combined_indices[tf.newaxis, tf.newaxis, :, :]
    combined_indices = tf.tile(combined_indices, [b, n, 1, 1])
    flattened_indices = tf.reshape(combined_indices, [b, n, n*n])

    batch_indices = tf.range(b)[:, tf.newaxis, tf.newaxis]
    batch_indices = tf.tile(batch_indices, [1, n, n*n])
    position_indices = tf.range(n)[tf.newaxis, :, tf.newaxis]
    position_indices = tf.tile(position_indices, [b, 1, n*n])

    final_indices = tf.stack(
        [batch_indices, position_indices, flattened_indices], axis=-1)
    output_shape = [b, n, n]

    return tf.scatter_nd(final_indices, flattened_probabilities, shape=output_shape)


# TODO: comment
OPCODES = {
    'HLT': 0,
    'INP': 1,  # load from input register to accumulator #
    'OUT': 2,  # store to accumulator to output register #
    'LDA': 3,
    'SDA': 4,
    'ADD': 5,
    'MUL': 6,
    'SUB': 7,
    'BRA': 8,
    'BRZ': 9,
    'BRP': 10,
    'DAT': 11,  # no-op for data storage
    'BRI': 12,  # Indirect branch
    'LDI': 13,  # Indirect load #
    'SDI': 14,  # Indirect store #
}


class MachineModel:
    def __init__(self, memory_size, max_steps, noise_level=0.001, mode="linear", debug=False):  # mode: linear, softmax
        self.num_ops = len(OPCODES)
        self.memory_size = memory_size
        self.max_steps = max_steps
        self.localization_rate = 0.001
        self.debug = debug

        self.mode = mode
        if mode == 'softmax':
            instructions_noise = tf.random.uniform(
                [memory_size, self.num_ops], minval=-1., maxval=1., dtype=tf.float32)
            memory_noise = tf.random.uniform(
                [memory_size, memory_size], minval=-1, maxval=1., dtype=tf.float32)
            position_noise = tf.random.uniform(
                [memory_size], minval=-1, maxval=1., dtype=tf.float32)

            self.instruction_ = tf.Variable(
                instructions_noise*noise_level,
                dtype=tf.float32,
                trainable=True
            )
            self.initial_memory_ = tf.Variable(
                memory_noise*noise_level,
                dtype=tf.float32,
                trainable=True
            )

            self.initial_position_ = tf.Variable(
                position_noise*noise_level,
                dtype=tf.float32,
                trainable=True
            )
        elif mode == 'linear':
            instructions = tf.random.uniform(
                [memory_size, self.num_ops], minval=(1.-noise_level), maxval=1., dtype=tf.float32)
            memory = tf.random.uniform(
                [memory_size, memory_size], minval=(1.-noise_level), maxval=1., dtype=tf.float32)
            position = tf.random.uniform(
                [memory_size], minval=(1.-noise_level), maxval=1., dtype=tf.float32)

            self.instruction_ = tf.Variable(
                instructions,
                dtype=tf.float32,
                trainable=True
            )
            self.initial_memory_ = tf.Variable(
                memory,
                dtype=tf.float32,
                trainable=True
            )

            self.initial_position_ = tf.Variable(
                position,
                dtype=tf.float32,
                trainable=True
            )
        else:
            assert (False)

    def load(self, instruction, memory, position):
        self.instruction_ = tf.Variable(
            instruction, dtype=tf.float32, trainable=True)
        self.initial_memory_ = tf.Variable(
            memory, dtype=tf.float32, trainable=True)
        self.initial_position_ = tf.Variable(
            position, dtype=tf.float32, trainable=True)

    def step(self, inputs):
        initial_position, initial_accumulator, initial_memory, initial_output, initial_input = inputs
        tf.debugging.assert_non_negative(initial_position)
        tf.debugging.assert_non_negative(initial_accumulator)
        tf.debugging.assert_non_negative(self.instruction)

        batch_size = initial_position.shape[0]

        accumulators = []

        halt_position = tf.expand_dims(
            self.instruction[:, :, OPCODES['HLT']]*initial_position, axis=-1)
        halt_accumulator = initial_accumulator*halt_position
        accumulators.append((halt_accumulator, halt_position))

        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['DAT']]*initial_position, axis=-1))
        accumulator = increment_program_counter(initial_accumulator)*position
        accumulators.append((accumulator, position))

        # loading data
        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['INP']]*initial_position, axis=-1))
        accumulator = increment_program_counter(
            tf.einsum('bij,bjk->bik', initial_memory, initial_input))*position
        accumulators.append((accumulator, position))

        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['LDA']]*initial_position, axis=-1))
        accumulator = increment_program_counter(
            tf.einsum('bij,bjk->bik', initial_memory, initial_memory))*position
        accumulators.append((accumulator, position))

        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['LDI']]*initial_position, axis=-1))
        accumulator = increment_program_counter(tf.einsum('bij,bja,bak->bik',
                                                          initial_memory, initial_memory, initial_memory))*position
        accumulators.append((accumulator, position))

        # arithmetic
        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['ADD']]*initial_position, axis=-1))
        accumulator = increment_program_counter(
            add(initial_memory, initial_accumulator))*position
        accumulators.append((accumulator, position))

        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['SUB']]*initial_position, axis=-1))
        accumulator = increment_program_counter(
            sub(initial_memory, initial_accumulator))*position
        accumulators.append((accumulator, position))

        position = increment_program_counter(tf.expand_dims(
            self.instruction[:, :, OPCODES['MUL']]*initial_position, axis=-1))
        accumulator = increment_program_counter(
            mul(initial_memory, initial_accumulator))*position
        accumulators.append((accumulator, position))

        # branching
        position = tf.expand_dims(tf.einsum(
            'bi,bij->bj', initial_position*self.instruction[:, :, OPCODES['BRA']], initial_memory), axis=-1)
        accumulator = tf.einsum('bi,bij,bik->bjk', initial_position *
                                self.instruction[:, :, OPCODES['BRA']], initial_memory, initial_accumulator)
        accumulators.append((accumulator, position))

        position = tf.expand_dims(tf.einsum('bi,bia,baj->bj', initial_position *
                                  self.instruction[:, :, OPCODES['BRI']], initial_memory, initial_memory), axis=-1)
        accumulator = tf.einsum('bi,bia,baj,bik->bjk', initial_position *
                                self.instruction[:, :, OPCODES['BRI']], initial_memory, initial_memory, initial_accumulator)
        accumulators.append((accumulator, position))

        accumulator_brz = p_brz(initial_accumulator)
        prob_brz_transition = tf.reduce_sum(accumulator_brz, axis=-1)
        prob_brz_and_position = initial_position * \
            self.instruction[:, :, OPCODES['BRZ']]
        position = tf.expand_dims(tf.einsum(
            'bi,bij->bj', prob_brz_and_position*prob_brz_transition, initial_memory), axis=-1)
        accumulator = tf.einsum(
            'bi,bij,bik->bjk', prob_brz_and_position, initial_memory, accumulator_brz)
        accumulators.append((accumulator, position))

        position = increment_program_counter(tf.expand_dims(
            prob_brz_and_position * (tf.ones(shape=[batch_size, self.memory_size]) - prob_brz_transition), axis=-1))

        accumulator = increment_program_counter(initial_accumulator - accumulator_brz) * \
            increment_program_counter(
                tf.expand_dims(prob_brz_and_position, axis=-1))
        accumulators.append((accumulator, position))

        accumulator_brp = p_brp(initial_accumulator)
        prob_brp_transition = tf.reduce_sum(accumulator_brp, axis=-1)
        prob_brp_and_position = initial_position * \
            self.instruction[:, :, OPCODES['BRP']]
        position = tf.expand_dims(tf.einsum(
            'bi,bij->bj', prob_brp_and_position * prob_brp_transition, initial_memory), axis=-1)
        accumulator = tf.einsum(
            'bi,bij,bik->bjk', prob_brp_and_position, initial_memory, accumulator_brp)
        accumulators.append((accumulator, position))

        position = increment_program_counter(tf.expand_dims(
            prob_brp_and_position * (tf.ones(shape=[batch_size, self.memory_size]) - prob_brp_transition), axis=-1))
        accumulator = increment_program_counter(initial_accumulator - accumulator_brp) * \
            increment_program_counter(
                tf.expand_dims(prob_brp_and_position, axis=-1))
        accumulators.append((accumulator, position))

        # saving data
        position_out = increment_program_counter(tf.expand_dims(
            initial_position*self.instruction[:, :, OPCODES['OUT']], axis=-1))
        output_output = tf.einsum('bi,bij,bik->bjk', initial_position *
                                  self.instruction[:, :, OPCODES['OUT']], initial_memory, initial_accumulator)
        output_output = tf.concat(
            [output_output, tf.zeros(shape=[batch_size, self.memory_size, 1])], axis=-1)
        accumulator = increment_program_counter(
            initial_accumulator)*position_out
        accumulators.append((accumulator, position_out))

        position_sda = increment_program_counter(tf.expand_dims(
            initial_position*self.instruction[:, :, OPCODES['SDA']], axis=-1))
        output_memory = tf.einsum('bi,bij,bik->bjk', initial_position *
                                  self.instruction[:, :, OPCODES['SDA']], initial_memory, initial_accumulator)
        accumulator = increment_program_counter(
            initial_accumulator)*position_sda
        accumulators.append((accumulator, position_sda))

        position_sdi = increment_program_counter(tf.expand_dims(
            initial_position*self.instruction[:, :, OPCODES['SDI']], axis=-1))
        output_memory += tf.einsum('bi,bia,baj,bik->bjk', initial_position *
                                   self.instruction[:, :, OPCODES['SDI']], initial_memory, initial_memory, initial_accumulator)
        accumulator = increment_program_counter(
            initial_accumulator)*position_sdi
        accumulators.append((accumulator, position_sdi))

        final_memory = (tf.ones(shape=(batch_size, self.memory_size)) - tf.reduce_sum(
            output_memory, axis=-1))[:, :, tf.newaxis]*initial_memory + output_memory
        final_output = (tf.ones(shape=(batch_size, self.memory_size)) - tf.reduce_sum(
            output_output, axis=-1))[:, :, tf.newaxis]*initial_output + output_output

        total = None
        norm = None
        i = 0
        for accumulator, position in accumulators:
            i += 1
            if total is None:
                total = accumulator
            else:
                total += accumulator

            if norm is None:
                norm = position
            else:
                norm += position

        final_accumulator = tf.where(
            norm > 1e-8, total/norm, tf.ones_like(initial_accumulator)/self.memory_size)

        final_position = tf.squeeze(norm, axis=-1)

        if self.debug:
            tf.debugging.assert_less(
                1.-tf.reduce_sum(initial_position, axis=-1), 1e-6)
            tf.debugging.assert_greater(
                1.-tf.reduce_sum(initial_position, axis=-1), -1e-6)

            tf.debugging.assert_less(
                1.-tf.reduce_sum(final_memory, axis=-1), 1e-6)
            tf.debugging.assert_less(
                1.-tf.reduce_sum(final_accumulator, axis=-1), 1e-6)
            tf.debugging.assert_less(
                1.-tf.reduce_sum(final_output, axis=-1), 1e-6)
            tf.debugging.assert_less(
                1.-tf.reduce_sum(final_position, axis=-1), 1e-6)
            tf.debugging.assert_greater(
                1.-tf.reduce_sum(final_memory, axis=-1), -1e-6)
            tf.debugging.assert_greater(
                1.-tf.reduce_sum(final_accumulator, axis=-1), -1e-6)
            tf.debugging.assert_greater(
                1.-tf.reduce_sum(final_output, axis=-1), -1e-6)
            tf.debugging.assert_greater(
                1.-tf.reduce_sum(final_position, axis=-1), -1e-6)

        # take absolute value in case of small negative values due to precision errors
        final_memory = tf.math.abs(final_memory)
        final_accumulator = tf.math.abs(final_accumulator)
        final_output = tf.math.abs(final_output)
        final_position = tf.math.abs(final_position)

        final_memory = final_memory / \
            tf.reduce_sum(final_memory, axis=-1, keepdims=True)
        final_accumulator = final_accumulator / \
            tf.reduce_sum(final_accumulator, axis=-1, keepdims=True)
        final_output = final_output / \
            tf.reduce_sum(final_output, axis=-1, keepdims=True)
        final_position = final_position / \
            tf.reduce_sum(final_position, axis=-1, keepdims=True)

        return [final_position, final_accumulator, final_memory, final_output, initial_input]

    def call(self, inputs, training=False):
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        outputs = []

        if self.mode == 'softmax':
            self.instruction = tf.math.exp(self.instruction_)[tf.newaxis, :, :]
            self.initial_memory = tf.math.exp(self.initial_memory_)[
                tf.newaxis, :, :]
            self.initial_position = tf.math.exp(
                self.initial_position_)[tf.newaxis, :]
        elif self.mode == 'linear':
            self.instruction = self.instruction_[tf.newaxis, :, :]
            self.initial_memory = self.initial_memory_[tf.newaxis, :, :]
            self.initial_position = self.initial_position_[tf.newaxis, :]
        else:
            assert (False)

        self.instruction = self.instruction / \
            tf.reduce_sum(self.instruction, axis=-1, keepdims=True)
        self.instruction = tf.tile(self.instruction, [batch_size, 1, 1])
        self.initial_memory = self.initial_memory / \
            tf.reduce_sum(self.initial_memory, axis=-1, keepdims=True)
        self.initial_memory = tf.tile(self.initial_memory, [batch_size, 1, 1])
        self.initial_position = self.initial_position / \
            tf.reduce_sum(self.initial_position, axis=-1, keepdims=True)
        self.initial_position = tf.tile(self.initial_position, [batch_size, 1])

        position = self.initial_position

        np_accumulator = np.zeros(
            shape=[batch_size, self.memory_size, self.memory_size])
        np_accumulator[:, :, 0] = np.ones_like(np_accumulator[:, :, 0])
        accumulator = tf.constant(np_accumulator, dtype=tf.float32)

        memory = self.initial_memory
        np_output = np.zeros(
            shape=[batch_size, self.memory_size, self.memory_size+1])
        np_output[:, :, self.memory_size] = np.ones_like(
            np_output[:, :, self.memory_size])
        output = tf.constant(np_output, dtype=tf.float32)

        input_ = tf.constant(tf.one_hot(inputs, depth=self.memory_size))

        for i in range(self.max_steps):
            position, accumulator, memory, output, _ = self.step(
                [position, accumulator, memory, output, input_])
            if training:
                outputs.append([output, position, accumulator, memory])
        if training:
            return outputs
        else:
            return [output, position, accumulator, memory]

    def compile(self, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics={"accuracy": argmax_accuracy, "sampled_accuracy": sampled_accuracy}):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def fit(self, X, y, batch_size, epochs=1, X_v=None, y_v=None):
        num_examples = X.shape[0]
        i = 0
        j = min(batch_size, num_examples)
        while i < num_examples:
            x_batch = X[i:j]
            y_true = y[i:j]
            i = j
            j = min(i+batch_size, num_examples)
            with tf.GradientTape() as tape:
                outputs = self.call(x_batch, training=True)
                loss = 0.0
                for step, step_output in enumerate(outputs):
                    if step == len(outputs)-1:
                        weight = 1.
                    else:
                        weight = 0./self.max_steps

                    y_pred, position, accumulator, memory = step_output
                    loss += weight * \
                        tf.reduce_mean(
                            self.loss(y_true, y_pred, from_logits=False))

                    loss += weight*self.localization_rate * \
                        (1.-tf.reduce_mean(position *
                         self.instruction[:, :, OPCODES['HLT']]))
                    loss += weight*self.localization_rate * \
                        (1. - tf.reduce_mean(self.initial_position_[0]))

            gradients = tape.gradient(
                loss, [self.initial_memory_, self.instruction_, self.initial_position_])
            print("loss", float(loss))

            self.optimizer.apply_gradients(
                zip(gradients, [self.initial_memory_, self.instruction_, self.initial_position_]))

            if self.mode == 'linear':
                self.initial_memory_.assign(tf.clip_by_value(
                    self.initial_memory_, 1e-3, 100000.))
                self.instruction_.assign(tf.clip_by_value(
                    self.instruction_, 1e-3, 100000.))
                self.initial_position_.assign(tf.clip_by_value(
                    self.initial_position_, 1e-3, 100000.))

            for metric in self.metrics:
                print(metric, float(self.metrics[metric](y_true, y_pred)))
