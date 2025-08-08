import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";

window.onload = async () => {
  //1.准备数据
  // 训练集
  const trainData = {
    sizeMB: [
      0.08, 9.0, 0.001, 0.1, 8.0, 5.0, 0.1, 6.0, 0.05, 0.5, 0.002, 2.0, 0.005,
      10.0, 0.01, 7.0, 6.0, 5.0, 1.0, 1.0,
    ],
    timeSec: [
      0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116,
      0.07, 0.289, 0.076, 0.744, 0.083, 0.56, 0.48, 0.399, 0.153, 0.149,
    ],
  };
  // 测试集
  const testData = {
    sizeMB: [
      5.0, 0.2, 0.001, 9.0, 0.002, 0.02, 0.008, 4.0, 0.001, 1.0, 0.005, 0.08,
      0.8, 0.2, 0.05, 7.0, 0.005, 0.002, 8.0, 0.008,
    ],
    timeSec: [
      0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.07, 0.375, 0.058, 0.136,
      0.052, 0.063, 0.183, 0.087, 0.066, 0.558, 0.066, 0.068, 0.61, 0.057,
    ],
  };

  const renderName = { name: "文件下载所需时间" };
  const trainValues = trainData.sizeMB.map((x, i) => ({
    x,
    y: trainData.timeSec[i],
  }));
  const testValues = testData.sizeMB.map((x, i) => ({
    x,
    y: testData.timeSec[i],
  }));
  const series = ["trainData", "testData"];
  const values = {
    values: [trainValues, testValues],
    series,
  };
  tfvis.render.scatterplot(renderName, values);
  // 2. 将数据转换为张量
  const trainTensors = {
    sizeMB: tf.tensor2d(trainData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(trainData.timeSec, [20, 1]),
  };
  const testTensors = {
    sizeMB: tf.tensor2d(testData.sizeMB, [20, 1]),
    timeSec: tf.tensor2d(testData.timeSec, [20, 1]),
  };
  // 3.构建模型
  const model = tf.sequential();
  //添加参数
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  // 添加损失函数 和优化器
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.01),
  });

  // sgd 指定为优化器 随机梯度下降
  // MSE meanSquaredError
  // 使用训练来拟合数据
  await model.fit(trainTensors.sizeMB, trainTensors.timeSec, {
    batchSize: 20,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: "训练过程" }, ["loss"]),
  });

  model.evaluate(testTensors.sizeMB, testTensors.timeSec).print()
};
