import * as tf from "@tensorflow/tfjs";
import * as render from "@tensorflow/tfjs-vis";
import { getData } from "./data";

window.onload = async () => {
  const data = getData(400,3);
  render.render.scatterplot(
    { name: "1" },
    {
      values: [
        data.filter((p) => p.label === 1),
        data.filter((p) => p.label === 0),
      ],
    }
  );

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 10,
    inputShape: [2],
    activation:'tanh'
  }))
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
  // 准备数据
  const inputs = tf.tensor(data.map((p) => [p.x, p.y]));
  const labels = tf.tensor(data.map((p) => p.label));
  await model.fit(inputs, labels, {
    epochs: 200,
    validationSplit: 0.2,
    batchSize: 320,
    callbacks: render.show.fitCallbacks(
      {
        name: "111",
      },
      ["loss", "val_loss"],
      { callbacks: ['onEpochEnd'] }
    ),
  });
};
