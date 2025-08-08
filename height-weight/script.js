import * as tfvis from "@tensorflow/tfjs-vis";
import * as tfjs from "@tensorflow/tfjs";

window.onload = async () => {
  const heights = [150, 160, 170];
  const weights = [50, 60, 70];
  tfvis.render.scatterplot(
    { name: "qiyue" },
    { values: weights.map((x, i) => ({ x, y: heights[i] })) },
    { xAxisDomain: [0, 100], yAxisDomain: [100, 200] }
  );

  const input = tfjs.tensor(heights).sub(150).div(20);
  const label = tfjs.tensor(weights).sub(50).div(20);
  const model = tfjs.sequential();
  model.add(tfjs.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: tfjs.losses.meanSquaredError,
    optimizer: tfjs.train.sgd(0.1),
  });
  await model.fit(input, label, {
    batchSize: 3,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]),
  });
  const output = model.predict(tfjs.tensor([180]).sub(150).div(20))
  console.log("🚀 ~ output:", `${output.mul(20).add(50).dataSync()[0]}`)
};
