import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data.js";

window.onload = async () => {
  const data = getData(400);
  console.log("🚀 ~ data:", data);
  tfvis.render.scatterplot(
    { name: "逻辑回归数据" },
    {
      values: [
        data.filter((p) => p.label === 1),
        data.filter((p) => p.label === 0),
      ],
    }
  );

  const model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 1, inputShape: [2], activation: "sigmoid" })
  );
  // Wiki.fast.ai
  model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) });
  // 1。转化tensor
  const inputs = tf.tensor(data.map((ponit) => [ponit.x, ponit.y]));
  console.log("🚀 ~ inputs:", inputs);
  const labels = tf.tensor(data.map((ponit) => ponit.label));
  console.log("🚀 ~ labels:", labels);
  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 200,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: "训练过程",
      },
      ["loss"]
    ),
  });

  window.predict = (form) => {
    console.log("🚀 ~ form:", form)
    const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value *1]]));
    alert(`预测结果：${pred.dataSync()[0]}`);
  };
};
