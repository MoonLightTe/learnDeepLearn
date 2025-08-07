import * as tf from "@tensorflow/tfjs";

const input = [1, 2, 3, 4];
const weight = [
  [1, 2, 3, 4],
  [2, 3, 4, 5],
  [3, 4, 5, 6],
  [4, 5, 6, 7],
];
const output = [0, 0, 0, 0];

for(let i = 0; i < weight.length; i++){
  for(let j = 0; j < input.length; j++){
    output[i] += weight[i][j] * input[j]
  }
}
console.log("🚀 ~ output:", output)

tf.tensor(weight).dot(tf.tensor(input)).print()



