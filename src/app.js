var model;

async function load() {
  document.querySelector('button').classList.add('disabled');
  document.querySelector('button').onclick = () => {};

  await startTfModel();

  await predictWithLastResults();

  enableUserInputPrediction();
}

async function predictWithLastResults() {
  const response = await fetch('../db/megasena.json');
  const results = (await response.json()).map((x) =>
    x.dezenas.map((i) => parseInt(i))
  );

  const prediction = model.predict(tf.tensor([results.slice(-20)])).dataSync()

  setResult(prediction);
}

function setResult(prediction) {
  console.log('Prediction: ', prediction);

  const sortByHighestProbability = (list) => {
    const listWithIndex = list.map((probability, index) => ({
      probability,
      number: index + 1,
    }));

    return listWithIndex.sort((a, b) => b.probability - a.probability);
  };

  const predResult = sortByHighestProbability(Array.from(prediction).slice(-60));

  console.log('Result: ', predResult);

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

  setResult(prediction);
}

async function startTfModel() {
  model = await tf.loadLayersModel('/src/assets/model/model.json', {
    strict: false,
  });

  console.log(model);
}
