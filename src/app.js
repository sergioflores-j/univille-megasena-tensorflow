// google.charts.load('current', { packages: ['corechart'] });

const N_DEZENAS = 6;
const N_NEURONIOS = 50;
const N_EPOCHS = 50;
const N_SORTEIOS = 40;
const BATCH_SIZE = 32;
var model;

const encode = (numbers) => {
  const encoded = [];

  for (let i = 0; i < 60; i++) {
    encoded[i] = numbers.includes(i + 1) ? 1 : 0;
  }

  return encoded;
};

const sumTotal = (numbers) => numbers.reduce((acc, curr) => acc + curr, 0);

const prepareTraining = (results) => {
  const TRAINING_LIMIT = 0.7 * results.length;
  const validationData = [...results];
  const trainData = validationData.splice(-TRAINING_LIMIT);
  // WITH ENCODE
  // const labels = trainData.map(encode);
  // WITH SUM
  const labels = trainData.map(sumTotal);

  console.log('trainData', trainData);
  console.log('labels', labels);

  return {
    inputs: tf.tensor(trainData),
    labels: tf.tensor(labels),
    validationInputs: tf.tensor(validationData),
    validationLabels: tf.tensor(validationData.map(sumTotal)),
  };
};

async function load() {
  document.querySelector('button').classList.add('disabled');
  document.querySelector('button').onclick = () => {};

  // ? Fazer na mao https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#2
  //  iimportar do keras:
  // Modelo base: https://github.com/JairoRotava/Colab_studies/blob/master/megasena_keras.ipynb
  // Como importar: https://www.tensorflow.org/js/tutorials/conversion/import_keras

  await startTfModel();
  // const { inputs, labels, validationInputs, validationLabels } = prepareTraining(results);

  // const history = await model.fit(inputs, labels, {
  //   epochs: N_EPOCHS,
  //   batchSize: BATCH_SIZE,
  //   shuffle: true,
  //   callbacks: tfvis.show.fitCallbacks(
  //     { name: 'Training Performance' },
  //     ['loss', 'mse'],
  //     { height: 200, callbacks: ['onEpochEnd'] }
  //   ),
  //   validationData: [validationInputs, validationLabels],
  // });

  console.log(history);

  await predictWithLastResults();

  // TODO: mostrar resultado em uma tabela
  // TODO: mostrar mais frequentes em uma tabela
  enableUserInputPrediction();
}

async function predictWithLastResults() {
  const response = await fetch('../db/megasena.json');
  const results = (await response.json()).map((x) =>
    x.dezenas.map((i) => parseInt(i))
  );

  // console.log(results);

  const prediction = model.predict(tf.tensor([results.slice(-20)])).dataSync()

  console.log('PREDICTION: ', prediction);
  setResult(prediction);
}

function setResult(prediction) {
  const sortByHighestProbability = (list) => {
    const listWithIndex = list.map((probability, index) => ({
      probability,
      number: index + 1,
    }));
    console.log('listWithIndex', listWithIndex);

    return listWithIndex.sort((a, b) => b.probability - a.probability);
  };

  const predResult = sortByHighestProbability(Array.from(prediction).slice(-60));

  console.log('Resposta: ', predResult);
  document.querySelector('#result').innerHTML = `
    <h3>Resultado:</h3>
    <p>No próximo sorteio escolha <b>${predResult
      .slice(0, 6)
      .map((i) => i.number)
      .join(', ')}</b></p>
    <table>
      <tr><th>Número</th><th>Probabilidade</th></tr>
      ${predResult
        .map(
          ({ probability, number }) =>
            `<tr><td>${number}</td><td>${probability * 100}</td></tr>`
        )
        .join('')}
    </table>
  `;
}

function enableUserInputPrediction() {
  document.querySelector('#p-content').innerHTML = `
    <label>Escolha uma combinação para verificar.</label>
    <div class="flex">
      ${[1, 2, 3, 4, 5, 6]
        .map(
          (i) => `
        <input type="number" id="num${i}" placeholder="${i}" value="${Math.ceil(
            Math.random() * 60
          )}" />
      `
        )
        .join('')} 
    </div>
  `;
  document.querySelector('button').classList.remove('disabled');
  document.querySelector('button').onclick = predict;
  document.querySelector('button').innerHTML = 'Predict';

  const resetPred = document.createElement('button');
  resetPred.onclick = predictWithLastResults;
  resetPred.innerHTML = 'Reset';
  document.querySelector('#buttons').appendChild(resetPred);
}

function predict() {
  const inputs = [
    document.querySelector('#num1').value,
    document.querySelector('#num2').value,
    document.querySelector('#num3').value,
    document.querySelector('#num4').value,
    document.querySelector('#num5').value,
    document.querySelector('#num6').value,
  ].map(Number);

  const prediction = model.predict(tf.tensor([[inputs]])).dataSync();
  console.log('Resposta: ', prediction);
  setResult(prediction);
}

async function startTfModel() {
  model = await tf.loadLayersModel('/src/assets/model/model.json', {
    strict: false,
  });
  console.log(model);

  // // Define o modelo de rede, detalhes em: https://www.tensorflow.org/js/guide/models_and_layers
  // model = tf.sequential({
  //   layers: [
  //     tf.layers.dense({
  //       units: 1,
  //       inputShape: [N_DEZENAS],
  //       activation: 'linear',
  //       useBias: true,
  //     }),
  //     tf.layers.dense({
  //       units: 1,
  //       useBias: true,
  //     }),

  //     // -- Activate below to check another type
  //     // https://github.com/tensorflow/tfjs-examples/blob/master/addition-rnn/index.js
  //     // tf.layers.simpleRNN({
  //     //   units: N_NEURONIOS,
  //     //   inputShape: [null, N_DEZENAS],
  //     //   // input_shape=[None, N_DEZENAS]
  //     //   activation: 'relu',
  //     //   returnSequences: true,
  //     // }),
  //     // tf.layers.dense({
  //     //   units: 60,
  //     //   activation: 'sigmoid',
  //     // }),
  //   ],
  // });

  // model.compile({
  //   optimizer: tf.train.adam(),
  //   loss: tf.losses.meanSquaredError,
  //   metrics: ['mse'],
  // });
  // // model.compile({
  // //   optimizer: tf.train.adam(),
  // //   loss: 'binaryCrossentropy',
  // //   metrics: ['accuracy'],
  // // });

  // console.log('Model Summary');
  // model.summary();
}
