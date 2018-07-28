import chai from 'chai';
import sinon from 'sinon';
import ms from 'modelscript';
import sinonChai from 'sinon-chai';
import chaiAsPromised from 'chai-as-promised';
import 'babel-polyfill';
import { MultipleLinearRegression, } from '../../index.mjs';
const expect = chai.expect;
const independentVariables = [ 'sqft', 'bedrooms',];
const dependentVariables = [ 'price', ];
let housingDataCSV;
let OriginalDataSet;
let DataSet;
let x_matrix;
let y_matrix;
let trainedMLR;
let trainedMLRModel;

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
describe('MultipleLinearRegression', function () {
  this.timeout(10000);
  before(async function () {
    housingDataCSV = await ms.csv.loadCSV('./test/mock/data/portland_housing_data.csv', {
      colParser: {
        sqft: 'number',
        bedrooms: 'number',
        price: 'number',
      },
    });
    /*
    housingdataCSV = [ 
      { sqft: 2104, bedrooms: 3, price: 399900 },
      { sqft: 1600, bedrooms: 3, price: 329900 },
      ...
      { sqft: 1203, bedrooms: 3, price: 239500 } 
    ] 
    */
    OriginalDataSet = new ms.DataSet(housingDataCSV);
    DataSet = new ms.DataSet(housingDataCSV);
    DataSet.fitColumns({
      columns: [
        'sqft',
        'bedrooms',
        'price',
      ].map(scaleColumnMap),
      returnData:false,
    });
    x_matrix = DataSet.columnMatrix(independentVariables); 
    y_matrix = DataSet.columnMatrix(dependentVariables);
    // const y_vector = ms.util.pivotVector(y_matrix)[ 0 ];// not used but just illustrative
    /* x_matrix = [
        [2014, 3],
        [1600, 3],
      ] 
      y_matrix = [
        [399900],
        [329900],
      ] 
      y_vector = [ 399900, 329900]
    */
    trainedMLR = new MultipleLinearRegression();
    trainedMLRModel = await trainedMLR.train(x_matrix, y_matrix);
  });
  describe('constructor', () => {
    it('should export a named module class', () => {
      const MLR = new MultipleLinearRegression();
      const MLRConfigured = new MultipleLinearRegression({ test: 'prop', });
      expect(MultipleLinearRegression).to.be.a('function');
      expect(MLR).to.be.instanceOf(MultipleLinearRegression);
      expect(MLRConfigured.settings.test).to.eql('prop');
    });
  });
  describe('async train', () => {
    it('should train data', async function () { 
      expect(trainedMLRModel).to.be.an('object');
      expect(trainedMLRModel.model).to.be.an('object');
    });
  });
  describe('calculate', () => {
    it('should return a tensorflow prediction', async function () { 
      const testSqft = DataSet.scalers.get('sqft').scale(1650);
      const testBedrooms = DataSet.scalers.get('bedrooms').scale(3);
      const input_x = [
        testSqft,
        testBedrooms,
      ]; // input_x: [ -0.4412732005944351, -0.2236751871685913 ]
      const predictionPromise = trainedMLR.calculate(input_x);
      predictionPromise.data()
        .then(predictions => {
          const scaledPrediction = predictions[ 0 ];
          expect(scaledPrediction).to.be.an('number');
          const prediction = DataSet.scalers.get('price').descale(scaledPrediction); // prediction: 293081.4643348962
          expect(prediction).to.be.closeTo(290000, 10000);
          // done();
          return true;
        })
        .catch(e => {
          throw e;
        });
      expect(predictionPromise).to.be.an('object');
    });
    it('should handle an array of predictions', () => {
      const input_x = [
        [
          DataSet.scalers.get('sqft').scale(4215),
          DataSet.scalers.get('bedrooms').scale(4),
        ], //549000
        [
          DataSet.scalers.get('sqft').scale(852),
          DataSet.scalers.get('bedrooms').scale(2),
        ], //179900
      ];
      const predictionPromise = trainedMLR.calculate(input_x);
      predictionPromise.data()
        .then(predictions => {
          const scaledPrediction = predictions[ 0 ];
          const descaledPredictions = predictions.map(DataSet.scalers.get('price').descale);
          expect(scaledPrediction).to.be.an('number');
          expect(predictions).to.have.lengthOf(input_x.length);
          expect(descaledPredictions[ 0 ]).to.be.closeTo(630000, 20000);
          expect(descaledPredictions[ 1 ]).to.be.closeTo(190000, 10000);
          return true;
        })
        .catch(e => {
          throw e;
        });
      expect(predictionPromise).to.be.an('object');
    });
  });
  describe('async predict', () => {
    it('should calculate and predict asynchronously', async function () {
      const testSqft = DataSet.scalers.get('sqft').scale(1650);
      const testBedrooms = DataSet.scalers.get('bedrooms').scale(3);
      const input_x = [
        testSqft,
        testBedrooms,
      ];
      const scaledPrediction = await trainedMLR.predict(input_x);
      console.log({scaledPrediction})
      // const descaledPredictions = scaledPrediction.map(DataSet.scalers.get('price').descale);
      // const prediction = descaledPredictions[ 0 ];
      // expect(trainedMLR.predict).to.be.a('function');
      // expect(prediction).to.be.closeTo(290000, 10000);
    });
  });
});