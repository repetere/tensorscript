import chai from 'chai';
// import sinon from 'sinon';
import * as ms from 'modelscript';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { LSTMMultivariateTimeSeries, } from '../../index.mjs';

const expect = chai.expect;
const independentVariables = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8',];
const dependentVariables = ['o1', ];
const ds = [
  [10, 20, 30, 40, 50, 60, 70, 80, 90, ],
  [11, 21, 31, 41, 51, 61, 71, 81, 91, ],
  [12, 22, 32, 42, 52, 62, 72, 82, 92, ],
  [13, 23, 33, 43, 53, 63, 73, 83, 93, ],
  [14, 24, 34, 44, 54, 64, 74, 84, 94, ],
  [15, 25, 35, 45, 55, 65, 75, 85, 95, ],
  [16, 26, 36, 46, 56, 66, 76, 86, 96, ],
  [17, 27, 37, 47, 57, 67, 77, 87, 97, ],
  [18, 28, 38, 48, 58, 68, 78, 88, 98, ],
  [19, 29, 39, 49, 59, 69, 79, 89, 99, ],
];
const columns = dependentVariables.concat(independentVariables);
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
  // const preInputShape = LSTMMultivariateTimeSeries.getInputShape(preddata.input);
  // console.log({ testData })
  // console.log({preInputShape})
  // console.log('preddata.input',preddata.input)
  // console.log('testData.x_matrix',testData.x_matrix)
  // const predictions = await preddata.model.predict(preddata.input);
  const predictions = await preddata.model.predict(testData.x_matrix);
  const predictions_unscaled = predictions.map(pred => DataSet.scalers.get('o1').descale(pred[ 0 ]), );
  const actuals_unscaled = testData.y_matrix.map(act => DataSet.scalers.get('o1').descale(act), );
  // console.log('predictions',predictions)
  // console.log('predictions_unscaled',predictions_unscaled)
  // console.log('actuals_unscaled',actuals_unscaled)
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
      actuals_unscaled, //actuals,
      predictions_unscaled, //estimates,
    ) * 100), //.toFixed(2)+'%',
  });
}

/** @test {LSTMMultivariateTimeSeries} */
describe('LSTMMultivariateTimeSeries', function () {
  this.timeout(120000);
  before(async function () {
    this.timeout(120000);
    /*
    csvData = [ 
      { o1: 10,
       i1: 20,
       i2: 30,
       i3: 40,
       i4: 50,
       i5: 60,
       i6: 70,
       i7: 80,
       i8: 90,
       field10: '' },
     { o1: 11,
       i1: 21,
       i2: 31,
       i3: 41,
       i4: 51,
       i5: 61,
       i6: 71,
       i7: 81,
       i8: 91,
       field10: '' },
      ...
      ];
    scaledData = [ 
      { o1: -1.3856406460551016,
       i1: -1.3856406460551016,
       i2: -1.3856406460551016,
       i3: -1.3856406460551016,
       i4: -1.3856406460551016,
       i5: -1.3856406460551016,
       i6: -1.3856406460551016,
       i7: -1.3856406460551016,
       i8: -1.3856406460551016,
       field10: '' },
     { o1: -1.0392304845413263,
       i1: -1.0392304845413263,
       i2: -1.0392304845413263,
       i3: -1.0392304845413263,
       i4: -1.0392304845413263,
       i5: -1.0392304845413263,
       i6: -1.0392304845413263,
       i7: -1.0392304845413263,
       i8: -1.0392304845413263,
       field10: '' },
      ...
    ]
    */
    csvData = await ms.csv.loadCSV('./test/mock/data/sample.csv');
    DataSet = new ms.DataSet(csvData);
    const scaledData = DataSet.fitColumns({
      columns: columns.map(scaleColumnMap),
      returnData:true,
    });
    //{ train_size: 16, test_size: 9 }
    train_size = parseInt(DataSet.data.length * 0.67);
    test_size = DataSet.data.length - train_size;
    train_x_data = DataSet.data.slice(0, train_size);
    test_x_data = DataSet.data.slice(train_size, DataSet.data.length);
    trainDataSet = new ms.DataSet(train_x_data);
    testDataSet = new ms.DataSet(test_x_data);
    x_matrix = trainDataSet.columnMatrix(columns); 
    x_matrix_test = testDataSet.columnMatrix(columns); 
    
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
    TSTS = new LSTMMultivariateTimeSeries({
      lookback: 1,
      features: 8,
      fit,
    });
    TSTSONE = new LSTMMultivariateTimeSeries({
      lookBack: 1,
      features: 8,
      fit,
    });
    evals = [
      {
        model: TSTS,
        modelname: 'TSTS',
      },
      {
        model: TSTSONE,
        modelname: 'TSTSONE',
      },
    ];
    return true;
  });
  /** @test {LSTMMultivariateTimeSeries#drop} */
  describe('static drop', () => {
    it('should drop matrix columns', () => {
      const data = [
        [0, 1, 2, 3,],
        [0, 1, 2, 3,],
        [0, 1, 2, 3,],
        [0, 1, 2, 3,],
      ];
      const droppedData = [
        [0, 2,],
        [0, 2,],
        [0, 2,],
        [0, 2,],
      ];
      const dropColumns = [1, 3,];
      const droppedFormatted = LSTMMultivariateTimeSeries.drop(data, dropColumns);
      expect(droppedFormatted).to.eql(droppedData);
    });
  });
  /** @test {LSTMMultivariateTimeSeries#getDropableColumns} */
  describe('static getDropableColumns', () => {
    const data = [
      [0,  1,  2,  3, 10, 11, 12, 13, ],
      [10, 11, 12, 13, 20, 21, 22, 23, ],
      [20, 21, 22, 23, 30, 31, 32, 33, ],
      [30, 31, 32, 33, 40, 41, 42, 43, ],
    ];
    it('should calculate dropable columns', () => {
      const n_in = 1;
      const n_out = 1;
      const features = 8;
      const drops = [10, 11, 12, 13, 14, 15, 16, 17, ];
      const droppableColumns = LSTMMultivariateTimeSeries.getDropableColumns(features, n_in, n_out);
      const droppableColumns1 = LSTMMultivariateTimeSeries.getDropableColumns(1, 1, 1);
      const droppableColumns2 = LSTMMultivariateTimeSeries.getDropableColumns(1, 3, 1);
      const droppableColumns3 = LSTMMultivariateTimeSeries.getDropableColumns(2, 3, 1);
      const droppableColumns4 = LSTMMultivariateTimeSeries.getDropableColumns(2, 1, 1);
      // console.log({ droppableColumns3,droppableColumns4 });
      expect(droppableColumns).to.eql(drops);
      expect(droppableColumns1).to.eql([3, ]);
      expect(droppableColumns2).to.eql([7, ]);
      expect(droppableColumns3).to.eql([10, 11, ]);
      expect(droppableColumns4).to.eql([4, 5, ]);
    });
    it('should throw an error if more that one future iteration', () => {
      expect(LSTMMultivariateTimeSeries.getDropableColumns.bind(null, 8, 1, 3)).to.throw(/Only 1 future/);
    });
  });
  /** @test {LSTMMultivariateTimeSeries#seriesToSupervised} */
  describe('static seriesToSupervised', () => {
    const originalData = [
      [0,  1,  2,  3, ],
      [10, 11, 12, 13, ],
      [20, 21, 22, 23, ],
      [30, 31, 32, 33, ],
      [40, 41, 42, 43, ],
    ];
    const data = [
      [0,  1,  2,  3, 10, 11, 12, 13, ],
      [10, 11, 12, 13, 20, 21, 22, 23, ],
      [20, 21, 22, 23, 30, 31, 32, 33, ],
      [30, 31, 32, 33, 40, 41, 42, 43, ],
    ];
    it('should generate supervised series', () => {
      const n_in = 1;
      const n_out = 1;
      const s1 = [
        [1, 2,],
        [11, 21,],
        [12, 22,],
        [13, 23,],
      ];
      const series = LSTMMultivariateTimeSeries.seriesToSupervised(originalData, n_in, n_out);
      const series1 = LSTMMultivariateTimeSeries.seriesToSupervised(s1, 3, n_out);
      expect(series).to.eql(data);
      expect(series1[ 0 ]).to.have.lengthOf(8);
      // expect(droppableColumns1).to.eql([3]);
      // expect(droppableColumns2).to.eql([7]);
      // expect(droppableColumns3).to.eql([10,11]);
      // expect(droppableColumns4).to.eql([4,5]);
    });
    it('should throw an error if more that one future iteration', () => {
      expect(LSTMMultivariateTimeSeries.seriesToSupervised.bind(null, originalData, 1, 3)).to.throw(/Only 1 future/);
    });
    it('should throw an error if no dependent variable supplied', () => {
      expect(LSTMMultivariateTimeSeries.seriesToSupervised.bind(null, [[1, ], [2, ], ], 1, 1)).to.throw(/Must include at least two columns/);
    });
  });
  /** @test {LSTMMultivariateTimeSeries#createDataset} */
  describe('static createDataset', () => {
    const lookback = 3;
    it('should return timeseries datasets', () => {
      const [datax, datay, features,] = LSTMMultivariateTimeSeries.createDataset(ds);
      const [datax2, datay2, features2,] = LSTMMultivariateTimeSeries.createDataset(ds, lookback);
      expect(datax).to.have.lengthOf(datay.length);
      expect(datax2).to.have.lengthOf(datay2.length);
      expect(datax[ 0 ]).to.have.lengthOf(features);
      expect(datax[ 0 ]).to.have.lengthOf(features*1+(1-1));
      expect(datax2[ 0 ]).to.have.lengthOf(features2*lookback+(lookback-1));
    });
  });
  /** @test {LSTMMultivariateTimeSeries#getTimeseriesShape} */
  describe('static getTimeseriesShape', () => {
    const [datax, datay, ] = LSTMMultivariateTimeSeries.createDataset(ds, 1);
    const [datax2, datay2, ] = LSTMMultivariateTimeSeries.createDataset(ds, 3);
    it('should calculate timeseries shape', () => {
      const tsShape = LSTMMultivariateTimeSeries.getTimeseriesShape.call({
        getInputShape: LSTMMultivariateTimeSeries.getInputShape,
        settings: {
          lookback: 1,
        },
      }, datax);
      const tsShape2 = LSTMMultivariateTimeSeries.getTimeseriesShape.call({
        getInputShape: LSTMMultivariateTimeSeries.getInputShape,
        settings: {
          lookback: 3,
        },
      }, datax2);
      expect(tsShape).to.eql([9,1,8 ]);
      expect(tsShape2).to.eql([7,3,26, ]);
    });
  });
  /** @test {LSTMMultivariateTimeSeries#getTimeseriesDataSet} */
  describe('static getTimeseriesDataSet', () => {
    const [datax, datay,] = LSTMMultivariateTimeSeries.createDataset(ds, 1);
    it('should return timeseries data', () => {
      const tsShape = LSTMMultivariateTimeSeries.getTimeseriesDataSet.call({
        getInputShape: LSTMMultivariateTimeSeries.getInputShape,
        settings: {
        },
      }, ds, 1);
      expect(tsShape.yShape).to.eql([9, 1,]);
      expect(tsShape.xShape).to.eql([9, 1,8,]);
      expect(tsShape.y_matrix).to.eql(datay);
      // expect(tsShape.x_matrix).to.eql(datax);
      // console.log({ tsShape, });
    });
  });
  /** @test {LSTMMultivariateTimeSeries#constructor} */
  describe('constructor', () => {
    it('should export a named module class', () => {
      const NN = new LSTMMultivariateTimeSeries();
      const NNConfigured = new LSTMMultivariateTimeSeries({ test: 'prop', });
      expect(LSTMMultivariateTimeSeries).to.be.a('function');
      expect(NN).to.be.instanceOf(LSTMMultivariateTimeSeries);
      expect(NNConfigured.settings.test).to.eql('prop');
    });
  });
  /** @test {LSTMMultivariateTimeSeries#predict} */
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
    it('should make single predictions', async () => {
      const testData = TSTSONE.getTimeseriesDataSet(x_matrix_test);
      const predictions = await TSTSONE.predict(testData.x_matrix[ 0 ]);
      expect(predictions).to.have.lengthOf(1);
      console.log({ predictions });
      return true;
    });
  });
  /** @test {LSTMMultivariateTimeSeries#train} */
  describe('async train', () => {
    it('should train a model with supplied test data', async () => {
      const testData = TSTSONE.getTimeseriesDataSet(x_matrix_test);
      const LSTMTS = new LSTMMultivariateTimeSeries({
        fit,features:8
      });
      const matrices = LSTMMultivariateTimeSeries.createDataset(x_matrix_test);
      const x = matrices[ 0 ];
      const y = matrices[ 1 ];
      const y_matrix = LSTMMultivariateTimeSeries.reshape(matrices[ 1 ], [ matrices[ 1 ].length, 1 ]);
      // console.log({x, y,y_matrix })

      await LSTMTS.train(x, y_matrix);
      const predictions = await TSTSONE.predict(testData.x_matrix[ 0 ]);
      const predictions_unscaled = predictions.map(pred => [DataSet.scalers.get('o1').descale(pred[ 0 ]),]);
      // console.log({ predictions_unscaled });
      expect(predictions).to.have.lengthOf(1);
      expect(LSTMTS.layers).to.be.a('object');
      return true;
    });
  });
  /** @test {LSTMMultivariateTimeSeries#generateLayers} */
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
      const LSTMTS = new LSTMMultivariateTimeSeries({ features: 8, fit, });
      const testData = LSTMTS.getTimeseriesDataSet(x_matrix_test);
      console.log('TSTSONE.layers', TSTSONE.layers);
      await LSTMTS.train(x_matrix, y_matrix, TSTSONE.layers,testData.x_matrix,testData.y_matrix);
      expect(LSTMTS.layers).to.be.a('object');
      return true;
    });
  });
});