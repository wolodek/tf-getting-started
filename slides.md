---
# try also 'default' to start simple
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: false
# some information about the slides, markdown enabled
info: |
  ## Getting started with TensorFlow
  Presentation for graduation seminar in Data Analysis,
  Faculty of Mathematics and Informatics,
  University of Lodz
# persist drawings in exports and build
drawings:
  persist: false
---

# TensorFlow

Getting started

---

# What is TensorFlow?

TensorFlow is a free and open-source software library for machine learning and artificial intelligence

- 🧠 **Developed by Google Brain** - Build for internal Google use in research and production, open-sourced in 2019
- 🐍 **Support most popular languages** - TensorFlow provides a stable Python and C++ API, with non-guaranted backward compatibile API for many others languages
- 🧑‍💻 **Developer friendly** - straightforward API, which enables creating even complicated ML models with ease 
- 👨‍🏭 **Production ready** - with all it's variants (TFX, TF Lite, TF.js) allows robust ML production no matter what platform you use
- 🌍 **No setup required** - Google Colab allows creating simple TF models in free, cloud environment
- 📖 **Free courses and resources** - Google and communicty created many great courses to help newcomers start with TF

---

# TensorFlow versions


- TensorFlow - core platform and library for machine learning
- TensorFlow.js - web implementation, allow models to predict on client's site (in the browser)
- TensorFlow Extended (TFX) - provide componends for building end-to-end production, ex. loading,validating, tuning data.
- TensorFlow Lite (TFLite) - API for mobile and embeddes devices


---
layout: section
---

# Getting started with TensorFlow

---

# Library structure

<img src="/tf.png"/> 

<div v-click="1" class="users"> End users </div>
<arrow v-click="1" x1="950" y1="140" x2="730" y2="140" color="#564" width="3" arrowSize="1" />
<arrow v-click="1" x1="950" y1="140" x2="800" y2="190" color="#564" width="3" arrowSize="1" />

<div v-click="2" class="researchers"> ML Researchers </div>
<arrow v-click="2" x1="950" y1="250" x2="645" y2="250" color="red" width="3" arrowSize="1" />

<style >
.users {
  position: absolute;
  top: 113px;
  left: 873px;
}

.researchers {
  position: absolute;
  top: 224px;
  left: 825px;
}
</style>

---

# Keras

Keras is a deep learning API writtent in Python, developed with focus on enabling fast experimentation.

- Simple -- but not simplistic. Keras reduces developer cognitive load to free you to focus on the parts of the problem that really matter.
- Flexible -- Keras adopts the principle of progressive disclosure of complexity: simple workflows should be quick and easy, while arbitrarily advanced workflows should be possible via a clear path that builds upon what you've already learned.
- Powerful -- Keras provides industry-strength performance and scalability: it is used by organizations and companies including NASA, YouTube, or Waymo.

---

# Tensorflow 2.X - Imports, Input and Output data

```python {1|2|4-8|10-14|all}
import tensorflow as tf
import numpy as np

# Input data
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

# Output data
y = np.array([[0],
              [1],
              [1],
              [0]], dtype=np.float32)
```

[1]: [Numpy UltraQuick Tutorial](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=numpy_tf2-colab&hl=pl)

<style>
.footnotes-sep {
  @apply mt-20 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

---

# Tensorflow 2.X - Creating Model

<br>

Keras let us create models using Model or Sequential Class.

Sequential is the simplest model, with linear stack of layers, connectd sequentially 1-> 2 -> 3

Model is for more sophisticated use cases, for example connections across multiple layers 1 <-> 3
```python {all}
# Create simple Sequential Model
model = tf.keras.models.Sequential()
```

[1]: [More in-depth difference between Model and Sequential](https://stackoverflow.com/a/66880334)

[2]: [Example of advancel Model usage](https://towardsdatascience.com/model-sub-classing-and-custom-training-loop-from-scratch-in-tensorflow-2-cc1d4f10fb4e)


<style>
.footnotes-sep {
  @apply mt-20 opacity-10;
}
.footnotes {
  @apply text-sm opacity-75;
}
.footnote-backref {
  display: none;
}
</style>

---

# Tensorflow 2.X - Layers, Activations

<br>

Keras provides some built-in layers and activation functions

```python {all}
tf.keras.activations.relu # Relu activation functions
tf.keras.activations.sigmoid # Sigmoid activation function
tf.keras.activations.tanh # Tanh activation function

tf.keras.layers.Dense # Regular, densely-connected NN layer
tf.keras.layers.Conv1d # 1D convolution layer
tf.keras.layers.dropout # Applies Dropout to the input, to prevent overfitting

# Example
tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid) # 1 Neuron layer with sigmoid activation function
```

---

# Tensorflow 2.X - Compile, train and predict

<br>

### First, we need to configure the model for training [^1]
```python {all}
model.compile(
    optimizer='Adam', # optimizer name or instance
    loss="mse", # Mean squared error
    metrics=["mae", "acc"] # Loss function - Mean Absolute Error, Accuracy
)
```

### Learning [^2]

```python {all}
model.fit(
    x=None, # Input data 
    y=None, # Target data
    batch_size=None, # Number of samples per gradient update
    epochs=1, # Number of epochs to train the model.
    shuffle=True, # Whether to shuffle the training data before each epoch
)
```

[^1]: [All compile options](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile)

[^2]: [More about fitting](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit)


---
layout: section
---

# XOR example

---

# XOR TF implementation - building model

```python {all}
import tensorflow as tf
import numpy as np

# Examples
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
# Labels
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(2,
  activation=tf.keras.activations.sigmoid,
  kernel_initializer=tf.initializers.Constant(0.5)))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

# Compile the model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
  loss=tf.keras.losses.MeanSquaredError(),
  metrics=['mse', 'binary_accuracy'))

model.fit(x, y, batch_size=1, epochs=500)
```

---

# XOR TF Implementation - prediction

```python
predictions = model.predict(x)
print(predictions)

#[[0.01753041]
# [0.9767829 ]
# [0.97674406]
# [0.04294848]]
```


---
layout: section
---

# TensorFlow.js

---

# Text Toxicity
<Suspense>
<TextToxicity />
</Suspense>

---

# Text Toxicity implementation - whole component
```ts {all|1-2,11,22}
import '@tensorflow/tfjs';
import { load } from '@tensorflow-models/toxicity'
import type { ToxicityClassifier } from '@tensorflow-models/toxicity';
import { ref, watch } from 'vue';
import type { Ref } from 'vue';
import { debounce } from 'lodash';
import { ClassificationResult } from './models'

const examples = [...] // small censorship 🚫
const treshold: Ref<number> = ref(0.9)
let model: ToxicityClassifier = await load(treshold.value, []);
const input: Ref<string> = ref('');
const result: Ref<ClassificationResult[]> = ref([]);
watch(treshold, debounce(loadModel, 500))

async function loadModel(treshold: number) {
  model = await load(treshold, [])
}

async function predict() {
  result.value = [];
  result.value = await model.classify([input.value]);
}
```

---

# Learn More

[Documentations](https://sli.dev) · [GitHub](https://github.com/slidevjs/slidev) · [Showcases](https://sli.dev/showcases.html)


---

# Sources


- [TensorFlow Docs](https://www.tensorflow.org/api/stable)
- [TensorFlow Guides](https://www.tensorflow.org/learn)
- [Keras Docs](https://keras.io/api/)
- [Google Developers - Introduction to TensorFlow]()
- [TF Model Class usage](https://developers.google.com/machine-learning/crash-course/first-steps-with-tensorflow/toolkit)
- [Stackoverflow](https://stackoverflow.com/questions/66879748/what-is-the-difference-between-tf-keras-model-and-tf-keras-sequential/66880334#66880334)
- [XOR](https://colab.research.google.com/drive/1gv-z-C9TpKAsnAyBYLmwRvk6NaAtlr_C#scrollTo=i5U1CnatZDk1)
- [Toxicity classifier](https://github.com/tensorflow/tfjs-models/tree/master/toxicity)