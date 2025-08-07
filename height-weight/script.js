import * as tfvis from "@tensorflow/tfjs-vis";
import * as tfjs from "@tensorflow/tfjs";

window.onload = () => {
  const heights = [150, 160, 170];
  const weights = [50, 60, 70];
  tfvis.render.scatterplot(
    { name: "qiyue" },
    { values: weights.map((x, i) => ({ x, y: heights[i] })) },
    { xAxisDomain: [0, 100], yAxisDomain: [100, 200] }
  );
};
