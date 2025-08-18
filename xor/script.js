import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import { getData } from "./data.js";

window.onload = async () => {
  const data = getData(400);
  tfvis.render.scatterplot(
    {
      name: "xor",
    },
    {
      values: [
        data.filter((p) => p.label == 1),
        data.filter((p) => p.label == 0),
      ],
    }
  );
  // 定义模型结构
  const model = tf.sequential();
  // 神经网络的激活函数的作用？
  model.add(
    tf.layers.dense({
      units: 4,
      inputShape: [2],
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "sigmoid",
    })
  );

  model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1),
  });

  const input = tf.tensor(data.map((p) => [p.x, p.y]));
  const label = tf.tensor(data.map((p) => p.label));

  await model.fit(input, label, {
    epochs: 100,
    batchSize: 400,
    callbacks: tfvis.show.fitCallbacks(
      {
        name: "1",
      },
      ["loss"]
    ),
  });

  window.predict = (form) => {
    console.log("🚀 ~ form:", form);
    const pred = model.predict(
      tf.tensor([[form.x.value * 1, form.y.value * 1]])
    );
    alert(`预测结果：${pred.dataSync()[0]}`);
  };
};
