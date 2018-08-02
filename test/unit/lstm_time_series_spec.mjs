import chai from 'chai';
// import sinon from 'sinon';
import * as ms from 'modelscript';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { LSTMTimeSeries, } from '../../index.mjs';
const expect = chai.expect;
const independentVariables = [
  'Passengers',
];
const dependentVariables = [
  'Passengers',
];
const columns = independentVariables;//.concat(dependentVariables);
let csvData;
let DataSet;
let x_matrix;
let y_matrix;
let train_size;
let test_size;
let train_x_data;
let test_x_data;
let trainDataSet;
let testDataSet;
let x_matrix_test;
let TSTS;
let TSTSStateful;
let TSTSONE;
let TSTSStatefulONE;
let accuracyTest = {};
let evals;
const ds = [
  [ 1, ], [ 2, ], [ 3, ], [ 4, ], [ 5, ], [ 6, ], [ 7, ], [ 8, ], [ 9, ], [ 10, ],
];

const fit= {
  epochs: 10,
  batchSize: 1,
};

chai.use(sinonChai);
chai.use(chaiAsPromised);
function scaleColumnMap(columnName) {
  return {
    name: columnName,
    options: {
      strategy: 'scale',
      scaleOptions: {
        strategy:'standard',
      },
    },
  };
}
async function getModelAccuracy(preddata){
  const m = await preddata.model.train(x_matrix);
  const testData = preddata.model.getTimeseriesDataSet(x_matrix_test);
  // const preInputShape = LSTMTimeSeries.getInputShape(preddata.input);
  // console.log({ testData })
  // console.log({preInputShape})
  // console.log('preddata.input',preddata.input)
  // console.log('testData.x_matrix',testData.x_matrix)
  // const predictions = await preddata.model.predict(preddata.input);
  const predictions = await preddata.model.predict(testData.x_matrix);
  const predictions_unscaled = predictions.map(pred => [DataSet.scalers.get('Passengers').descale(pred[ 0 ]),]);
  const actuals_unscaled = testData.y_matrix.map(act => [DataSet.scalers.get('Passengers').descale(act[ 0 ]),]);
  // let results = ms.DataSet.reverseColumnMatrix({
  //   vectors: predictions_unscaled,
  //   labels: dependentVariables,
  // });
  return({
    model: preddata.modelname,
    // predictions,
    // actuals_unscaled,
    // predictions_unscaled,
    // results,
    accuracy: (ms.util.rSquared(
      ms.util.pivotVector(actuals_unscaled)[ 0 ], //actuals,
      ms.util.pivotVector(predictions_unscaled)[ 0 ], //estimates,
    ) * 100), //.toFixed(2)+'%',
  });
}
/** @test {LSTMTimeSeries} */
describe('LSTMTimeSeries', function () {
  this.timeout(120000);
  before(async function () {
    this.timeout(120000);
    /*
    csvData = [ 
      { Month: '1949-01', Passengers: 112 },
      { Month: '1949-02', Passengers: 118 },
      { Month: '1949-03', Passengers: 132 },
      { Month: '1949-04', Passengers: 129 },
      ...
      ];
    scaledData = [ 
      { Month: '1949-01', Passengers: -1.4028822039369186 },
      { Month: '1949-02', Passengers: -1.3528681653893018 },
      ...
    ]
    */
    csvData = await ms.csv.loadCSV('./test/mock/data/international-airline-passengers-no_footer.csv');
    DataSet = new ms.DataSet(csvData);
    const scaledData = DataSet.fitColumns({
      columns: columns.map(scaleColumnMap),
      returnData:false,
    });
    //{ train_size: 96, test_size: 48 }
    train_size = parseInt(DataSet.data.length * 0.67);
    test_size = DataSet.data.length - train_size;
    train_x_data = DataSet.data.slice(0, train_size);
    test_x_data = DataSet.data.slice(train_size, DataSet.data.length);
    trainDataSet = new ms.DataSet(train_x_data);
    testDataSet = new ms.DataSet(test_x_data);
    x_matrix = trainDataSet.columnMatrix(independentVariables); 
    x_matrix_test = testDataSet.columnMatrix(independentVariables); 
    
    /*
    x_matrix = [ 
      [ -1.4028822039369186 ],
      [ -1.3528681653893018 ],
      [ -1.2361687421115288 ],
      ...
    ]; 
    x_matrix_test: = [ 
      [ 0.2892594335907897 ],
      [ 0.17256001031301674 ],
      [ 0.6310220303328392 ],
      ...
    ]; 
    */
    TSTS = new LSTMTimeSeries({
      lookBack: 3,
      fit,
    });
    TSTSStateful = new LSTMTimeSeries({
      lookBack: 3,
      stateful: true,
      fit,
    });
    TSTSONE = new LSTMTimeSeries({
      lookBack: 1,
      fit,
    });
    TSTSStatefulONE = new LSTMTimeSeries({
      lookBack: 1,
      stateful: true,
      fit,
    });
    evals = [
      {
        model: TSTS,
        modelname: 'TSTS',
      },
      {
        model: TSTSStateful,
        modelname: 'TSTSStateful',
      },
      {
        model: TSTSONE,
        modelname: 'TSTSONE',
      },
      {
        model: TSTSStatefulONE,
        modelname: 'TSTSStatefulONE',
      },
    ];
    return true;
  });
  /** @test {LSTMTimeSeries#createDataset} */
  describe('static createDataset', () => {
    const lookback = 3;
    it('should return timeseries datasets', () => {
      const [datax, datay, ] = LSTMTimeSeries.createDataset(ds);
      const [ datax2, datay2, ] = LSTMTimeSeries.createDataset(ds, lookback);
      expect(datax).to.have.lengthOf(datay.length);
      expect(datax2).to.have.lengthOf(datay2.length);
      expect(datax[ 0 ]).to.have.lengthOf(1);
      expect(datax2[ 0 ]).to.have.lengthOf(lookback);
    });
  });
  /** @test {LSTMTimeSeries#getTimeseriesShape} */
  describe('static getTimeseriesShape', () => {
    const [datax, datay,] = LSTMTimeSeries.createDataset(ds, 3);
    it('should calculate timeseries shape', () => {
      const tsShape = LSTMTimeSeries.getTimeseriesShape.call({
        getInputShape: LSTMTimeSeries.getInputShape,
        settings: {
          timeSteps: 1,
          mulitpleTimeSteps:false,
        },
      }, datax);
      const tsShape2 = LSTMTimeSeries.getTimeseriesShape.call({
        getInputShape: LSTMTimeSeries.getInputShape,
        settings: {
          timeSteps: 1,
          mulitpleTimeSteps:true,
        },
      }, datax);
      const tsShape3 = LSTMTimeSeries.getTimeseriesShape.call({
        getInputShape: LSTMTimeSeries.getInputShape,
        settings: {
          timeSteps: 1,
          stateful:true,
        },
      }, datax);
      expect(tsShape).to.eql([6, 1, 3,]);
      expect(tsShape2).to.eql([6, 3, 1,]);
      expect(tsShape3).to.eql([6, 3, 1,]);
    });
  });
  /** @test {LSTMTimeSeries#getTimeseriesDataSet} */
  describe('static getTimeseriesDataSet', () => {
    const [datax, datay,] = LSTMTimeSeries.createDataset(ds, 3);
    it('should return timeseries data', () => {
      const tsShape = LSTMTimeSeries.getTimeseriesDataSet.call({
        getInputShape: LSTMTimeSeries.getInputShape,
        settings: {
          timeSteps: 1,
          mulitpleTimeSteps:true,
        },
      }, ds, 3);
      expect(tsShape.yShape).to.eql([6, 1,]);
      expect(tsShape.xShape).to.eql([6, 3,]);
      expect(tsShape.y_matrix).to.eql(datay);
      // expect(tsShape.x_matrix).to.eql(datax);
      // console.log({ tsShape, });
    });
  });
  /** @test {LSTMTimeSeries#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const NN = new LSTMTimeSeries();
      const NNConfigured = new LSTMTimeSeries({ test: 'prop', });
      expect(LSTMTimeSeries).to.be.a('function');
      expect(NN).to.be.instanceOf(LSTMTimeSeries);
      expect(NNConfigured.settings.test).to.eql('prop');
    });
  });
  /** @test {LSTMTimeSeries#predict} */
  describe('async predict', () => {
    it('should allow for stateless predictions with one step time windows', async () => {
      const accr = await getModelAccuracy({ model: TSTSONE, modelname: 'TSTSONE', });
      // console.log({ accr });
      expect(accr.accuracy).to.be.ok;
      return true;
    });
    it('should allow for stateless predictions with multiple step time windows', async () => {
      const accr = await getModelAccuracy({ model: TSTS, modelname: 'TSTSONE', });
      // console.log({ accr, });
      expect(accr.accuracy).to.be.ok;
      return true;
    });
    it('should make stateful predictions', async () => {
      const accr = await getModelAccuracy({ model: TSTSStateful, modelname: 'TSTSStateful', });
      const accr2 = await getModelAccuracy({ model: TSTSStatefulONE, modelname: 'TSTSStatefulONE', });
      // console.log({ accr, accr2 });
      expect(accr.accuracy).to.be.ok;
      expect(accr2.accuracy).to.be.ok;
      return true;
    });
    it('should make single predictions', async () => {
      const testData = TSTSONE.getTimeseriesDataSet(x_matrix_test);
      const predictions = await TSTSONE.predict(testData.x_matrix[ 0 ]);
      expect(predictions).to.have.lengthOf(1);
      // console.log({ predictions });
      return true;
    });
  });
  /** @test {LSTMTimeSeries#train} */
  describe('async train', () => {
    it('should train a model with supplied test data', async () => {
      const testData = TSTSONE.getTimeseriesDataSet(x_matrix_test);
      const LSTMTS = new LSTMTimeSeries({ layerPreference: 'custom', fit, });
      const matrices = LSTMTimeSeries.createDataset(x_matrix_test);
      const x = matrices[ 0 ];
      const y = matrices[ 1 ];
      await LSTMTS.train(x, y);
      const predictions = await TSTSONE.predict(testData.x_matrix[ 0 ]);
      const predictions_unscaled = predictions.map(pred => [DataSet.scalers.get('Passengers').descale(pred[ 0 ]),]);
      console.log({ predictions_unscaled });
      expect(predictions).to.have.lengthOf(1);
      expect(LSTMTS.layers).to.be.a('object');
      return true;
    });
  });
  /** @test {LSTMTimeSeries#generateLayers} */
  describe('generateLayers', () => {
    // it('should generate a classification network', async () => {
    //   const predictions = await nnClassification.predict(input_x);
    //   const answers = await nnClassification.predict(input_x, {
    //     probability:false,
    //   });
    //   const shape = nnClassification.getInputShape(predictions);
    //   // console.log('nnClassification.layers', nnClassification.layers);
    //   // console.log({
    //   //   predictions,
    //   //   // probabilities,
    //   //   answers,
    //   //   // results,
    //   //   shape,
    //   // });
    //   expect(predictions).to.have.lengthOf(input_x.length);
    //   expect(nnClassification.layers).to.have.lengthOf(2);
    //   expect(shape).to.eql([5, 3,]);
    //   expect(answers[ 0 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
    //   // expect(answers[ 1 ]).to.eql(encodedAnswers[ 'Iris-virginica' ]);
    //   // expect(answers[ 2 ]).to.eql(encodedAnswers[ 'Iris-versicolor' ]);
    //   // expect(answers[ 3 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
    //   // expect(answers[ 4 ]).to.eql(encodedAnswers[ 'Iris-setosa' ]);
    //   return true;
    // });
    it('should generate a network from layers', async () => { 
      const LSTMTS = new LSTMTimeSeries({ layerPreference: 'custom', fit, });
      console.log('TSTSONE.layers', TSTSONE.layers);
      await LSTMTS.train(x_matrix, y_matrix, TSTSONE.layers);
      expect(LSTMTS.layers).to.be.a('object');
      return true;
    });
  });
});