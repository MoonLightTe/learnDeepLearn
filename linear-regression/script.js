import * as tfvis from "@tensorflow/tfjs-vis";
import * as tfjs from "@tensorflow/tfjs";

window.onload = async () => {
  const xs = [1, 2, 3, 4];
  const ys = [1, 3, 5, 7];
  tfvis.render.scatterplot(
    { name: "qiyue" },
    { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    { xAxisDomain: [0, 10], yAxisDomain: [0, 10] }
  );

  const model = tfjs.sequential();
  model.add(tfjs.layers.dense({ units: 1, inputShape: [1] }));
  // 损失函数和优化器
  model.compile({
    loss: tfjs.losses.meanSquaredError,
    optimizer: tfjs.train.sgd(0.1),
  });

  const input = tfjs.tensor(xs);
  const label = tfjs.tensor(ys);
  await model.fit(input, label, {
    batchSize: 4,
    epochs: 1000,
    callbacks: tfvis.show.fitCallbacks({name: '训练过程'}, ['loss'],)
  });

  const output = model.predict(tfjs.tensor([5]))
  console.log("🚀 ~ output: x 为5的时候",`${output.dataSync()[0]}`)
  
};
