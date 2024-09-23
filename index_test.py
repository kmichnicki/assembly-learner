import unittest
import tensorflow as tf
import numpy as np
from index import MachineModel, OPCODES


class TestModel(unittest.TestCase):
    def test_HLT(self):
        # HLT
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = np.zeros(shape=[1, 5])
        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["HLT"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[0 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[0 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0 for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        new_position = y[0]
        self.assertEqual(float(new_position[0][0]), 1.)

    def test_INP(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["INP"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[0 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[0 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[1. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        acc = y[1]
        self.assertEqual(acc[0, 1, 1], 1)

    def test_OUT(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["OUT"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[1 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[2 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[1. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[3][0, 2, 1]), 1)

    def test_LDA(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["LDA"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 2]), 1)

    def test_SDA(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["SDA"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[4 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 4]), 1)

    def test_ADD(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')
        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["ADD"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[1 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 3]), 1)

        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')
        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["ADD"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[4 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 1]), 1)

    def testSUB(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["SUB"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[2 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[1 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 1]), 1)

    def testMUL(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["MUL"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[2 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 4]), 1)

    def testBRA(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["BRA"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[2 for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 2]), 1)

    def testBRZ(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["BRZ"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 2]), 1)

        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["BRZ"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[1. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 1]), 1)

    def testBRP(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["BRP"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 2]), 1)

        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["BRP"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[2. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 1]), 1)

    def testDAT(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["DAT"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[2. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 1]), 1)

    def testBRI(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["BRI"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[2. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+2) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[0][0, 4]), 1)

    def testLDI(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["LDI"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[1][0, 1, 3]), 1)

    def testSDI(self):
        model = MachineModel(memory_size=5, max_steps=5,
                             mode='linear')

        position = tf.one_hot(np.array([0]), depth=5)
        model.instruction = tf.one_hot(
            np.array([[OPCODES["SDI"] for i in range(5)]]), depth=len(OPCODES))
        accumulator = tf.one_hot(np.array([[1. for i in range(5)]]), depth=5)
        memory = tf.one_hot(np.array([[(i+1) % 5 for i in range(5)]]), depth=5)
        outputs = tf.one_hot(np.array([[5 for i in range(5)]]), depth=6)
        inputs = tf.one_hot(np.array([[0. for i in range(5)]]), depth=5)
        x = [position, accumulator, memory, outputs, inputs]
        y = model.step(x)

        self.assertEqual(float(y[2][0, 2, 1]), 1)


if __name__ == '__main__':
    unittest.main()
