<template>
<div class="flex flex-col">
  <div>
    Treshold: <input min="0" max="1" step="0.01" type="range" v-model="treshold"> {{treshold}} 
  </div>
  <div class="mt-2">
    Input: <input style="width: 93%" class="border" type="text" v-model="input">
  </div>
  <div class="mt-2">
    <button class="px-3 py-2 bg-cyan-500 hover:bg-cyan-600 text-white" @click="predict">Predict</button>
    <button v-for="(el, i) in examples" class="ml-3 px-3 py-2 bg-red-500 hover:bg-red-600 text-white" @click="input = el">Example {{i + 1}}</button>
  </div>
  <table class="" >
    <tr>
      <th>Label</th>
      <th>Match</th>
      <th>Probability</th>
    </tr>
    <tr v-for="(el, i) in result" :key="i">
      <td>{{el.label}}</td>
      <td :class=" el.results[0].match ? 'text-green-500' : 'text-red-500' ">{{el.results[0].match}}</td>
      <td>{{el.results[0].probabilities[ el.results[0].match ? 1 : 0]}}</td>
    </tr>
  </table>
  
</div>

</template>

<script setup lang="ts">
import '@tensorflow/tfjs';
import { load } from '@tensorflow-models/toxicity'
import type { ToxicityClassifier } from '@tensorflow-models/toxicity';
import { ref, watch } from 'vue';
import type { Ref } from 'vue';
import { debounce } from 'lodash';
import { ClassificationResult } from './models'
const examples = [
  `Ouch bad move mother fucker. i'm coming for you now., going to rape you in your sleep.`,
  `I will flay you alive, you fking faggot.`,
  'YOU SUCK',
  'You suck'
]
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
</script>

<style>

</style>