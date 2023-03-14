import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.device('/GPU:0')
else:
    tf.device('/device:CPU:0')

E0_PBP = 0.123
Kbi21 = 10e-3
Kbi31 = 8e-4
Kbi12 = 5e-3
Kbi32 = 10e-5
Kbi23 = 2e-3
Kbi13 = 13e-2


# a1, a2 ions with charge number 1
# a3 ion with charge number of 2
def pbp_equation(a1, a2, a3):
    e1 = []
    e2 = []
    e3 = []
    for a1n, a2n, a3n in zip(a1, a2, a3):
        e1.append(E0_PBP + 0.0256 * np.log(0.5 * (a1n + Kbi21 * a2n) +
                                         np.sqrt((0.5 * (Kbi21 * a2n + a1n)) ** 2 + Kbi31 * a3n)))
    for a1n, a2n, a3n in zip(a1, a2, a3):
        e2.append(E0_PBP + 0.0256 * np.log(0.5 * (a2n + Kbi12 * a1n) +
                                         np.sqrt((0.5 * (a2n + Kbi12 * a1n)) ** 2 + Kbi32 * a3n)))
    for a1n, a2n, a3n in zip(a1, a2, a3):
        e3.append(E0_PBP + 0.0256 * np.log(0.5 * (Kbi23 * a2n + Kbi13 * a1n) +
                                         np.sqrt((0.5 * (Kbi23 * a2n + Kbi13 * a1n)) ** 2 + a3n)))
    return e1, e2, e3


def generate_pbp_train_data():
    a1 = np.linspace(0, 2000, 2000)
    a2 = np.linspace(1000, 3600, 2000)
    a3 = np.flip(np.linspace(1000, 5000, 2000))
    e1, e2, e3 = pbp_equation(a1, a2, a3)
    x = np.stack([e1, e2, e3], axis=-1)
    y = np.stack([a1, a2, a3], axis=-1)
    x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    return tf.convert_to_tensor(x_train, dtype=tf.bfloat16), \
           tf.convert_to_tensor(x_validate, dtype=tf.bfloat16), \
           tf.convert_to_tensor(y_train, dtype=tf.bfloat16), \
           tf.convert_to_tensor(y_validate, dtype=tf.bfloat16)


def generate_pbp_test_data():
    a1 = np.linspace(10, 110, 200)
    a2 = np.linspace(1000, 2000, 200)
    a3 = np.flip(np.linspace(15, 300, 200))
    e1, e2, e3 = pbp_equation(a1, a2, a3)
    x = np.stack([e1, e2, e3], axis=-1)
    y = np.stack([a1, a2, a3], axis=-1)
    return tf.convert_to_tensor([x, y], dtype=tf.bfloat16)


if __name__ == "__main__":
    x_train, x_validate, y_train, y_validate = generate_pbp_train_data()
    x_test, y_test = generate_pbp_test_data()

    print(x_train, y_train)
    print(x_test, y_test)