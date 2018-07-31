# Class

## `DeepLearningClassification`

Deep Learning Classification with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `yShape: *`

### `xShape: *`

### `layers: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow classification model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

## `BaseNeuralNetwork`

Deep Learning with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `xShape: *`

### `yShape: *`

### `model: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

### `train(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>, x_text: Array<Array<number>>, y_text: Array<Array<number>>): Object`

Asynchronously trains tensorflow model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | array of model dense layer parameters |
| x_text | Array<Array<number>> |  | validation data independent variables |
| y_text | Array<Array<number>> |  | validation data dependent variables |

### `calculate(matrix: Array<Array<number>>|Array<number>, options: Object): {data: Promise}`

Predicts new dependent variables

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| matrix | Array<Array<number>>|Array<number> |  | new test independent variables |
| options | Object |  | model prediction options |

## `LogisticRegression`

Logistic Regression Classification with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `yShape: *`

### `xShape: *`

### `layers: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>, x_test: Array<Array<number>>, y_test: Array<Array<number>>)`

Adds dense layers to tensorflow classification model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |
| x_test | Array<Array<number>> |  | validation data independent variables |
| y_test | Array<Array<number>> |  | validation data dependent variables |

## `LSTMMultivariateTimeSeries`

Long Short Term Memory Multivariate Time Series with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `createDataset: *`

### `seriesToSupervised: *`

### `getDropableColumns: *`

### `drop: *`

### `getTimeseriesShape: *`

### `getTimeseriesDataSet: *`

### `yShape: *`

### `xShape: *`

### `layers: *`

### `model: *`

### `createDataset(dataset: Array<Array<number>, look_back: Number): [Array<Array<number>>,Array<number>]`

Creates dataset data

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| dataset | Array<Array<number> |  | array of values |
| look_back | Number |  | number of values in each feature |

### `drop(data: Array<Array<number>>, columns: Array<number>): Array<Array<number>>`

Drops columns by array index

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| data | Array<Array<number>> |  | data set to drop columns |
| columns | Array<number> |  | array of column indexes |

### `seriesToSupervised(data: Array<Array<number>>, n_in: number, n_out: number): Array<Array<number>>`

Converts data set to supervised labels for forecasting, the first column must be the dependent variable

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| data | Array<Array<number>> |  | data set |
| n_in | number |  | look backs |
| n_out | number |  | future iterations (only 1 supported) |

### `getDropableColumns(features: number, n_in: number, n_out: number): Array<number>`

Calculates which columns to drop by index

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| features | number |  | number of independent variables |
| n_in | number |  | look backs |
| n_out | number |  | future iterations (currently only 1 supported) |

### `getTimeseriesShape(x_timeseries: Array<Array<number>): Array<Array<number>>`

Reshape input to be [samples, time steps, features]

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_timeseries | Array<Array<number> |  | dataset array of values |

### `getTimeseriesDataSet(timeseries: *, look_back: *)`

Returns data for predicting values

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| timeseries | * |  |
| look_back | * |  |

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>, x_test: Array<Array<number>>, y_test: Array<Array<number>>)`

Adds dense layers to tensorflow classification model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |
| x_test | Array<Array<number>> |  | validation data independent variables |
| y_test | Array<Array<number>> |  | validation data dependent variables |

### `train(x_timeseries: *, y_timeseries: *, layers: *, x_test: *, y_test: *)`

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_timeseries | * |  |
| y_timeseries | * |  |
| layers | * |  |
| x_test | * |  |
| y_test | * |  |

## `LSTMTimeSeries`

Long Short Term Memory Time Series with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `createDataset: *`

### `getTimeseriesDataSet: *`

### `getTimeseriesShape: *`

### `yShape: *`

### `xShape: *`

### `layers: *`

### `model: *`

### `createDataset(dataset: Array<Array<number>, look_back: Number): [Array<Array<number>>,Array<number>]`

Creates dataset data

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| dataset | Array<Array<number> |  | array of values |
| look_back | Number |  | number of values in each feature |

### `getTimeseriesShape(x_timeseries: Array<Array<number>): Array<Array<number>>`

Reshape input to be [samples, time steps, features]

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_timeseries | Array<Array<number> |  | dataset array of values |

### `getTimeseriesDataSet(timeseries: *, look_back: *)`

Returns data for predicting values

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| timeseries | * |  |
| look_back | * |  |

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>, x_test: Array<Array<number>>, y_test: Array<Array<number>>)`

Adds dense layers to tensorflow classification model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |
| x_test | Array<Array<number>> |  | validation data independent variables |
| y_test | Array<Array<number>> |  | validation data dependent variables |

### `train()`

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |

### `calculate()`

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |

### `predict()`

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |

## `TensorScriptModelInterface`

Base class for tensorscript models

### `constructor(options: Object, customTF: Object, properties: {model:Object,tf:Object,})`

### `settings: Object`

### `model: Object`

### `tf: Object`

### `reshape: Function`

### `getInputShape: Function`

### `reshape(array: Array<number>, shape: Array<number>): Array<Array<number>>`

Reshapes an array

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| array | Array<number> |  | input array |
| shape | Array<number> |  | shape array |

### `getInputShape(matrix: Array<Array<number>>): Array<number>`

Returns the shape of an input matrix

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| matrix | Array<Array<number>> |  | input matrix |

### `train(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>): Object`

Asynchronously trains tensorflow model, must be implemented by tensorscript class

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |

### `calculate(matrix: Array<Array<number>>|Array<number>): {data: Promise}`

Predicts new dependent variables

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| matrix | Array<Array<number>>|Array<number> |  | new test independent variables |

### `loadModel(options: Object): Object`

Loads a saved tensoflow / keras model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| options | Object |  | tensorflow load model options |

### `predict(input_matrix: Array<Array<number>>|Array<number>, options.json: Boolean, options.probability: Boolean): Array<number>|Array<Array<number>>`

Returns prediction values from tensorflow model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| input_matrix | Array<Array<number>>|Array<number> |  | new test independent variables |
| options.json | Boolean | optional: true, default: true | return object instead of typed array |
| options.probability | Boolean | optional: true, default: true | return real values instead of integers |

## `DimensionError`

Create a range error with the message: 'Dimension mismatch (<actual size> != <expected size>)' (from math.js)

### `constructor()`

### `actual: *`

### `expected: *`

### `relation: *`

### `message: *`

### `name: *`

### `isDimensionError: *`

## `MultipleLinearRegression`

Mulitple Linear Regression with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `layers: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow regression model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

## `DeepLearningRegression`

Deep Learning Regression with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object,layerPreference:String}, properties: {model:Object,tf:Object,})`

### `layers: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow regression model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

# Function

## `pivotVector(vectors: Array[]): Array[]`

Returns an array of vectors as an array of arrays (from modelscript)

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| vectors | Array[] |  |

## `size(x: Array): Number[]`

Calculate the size of a multi dimensional array. This function checks the size of the first entry, it does not validate whether all dimensions match. (use function `validate` for that) (from math.js)

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x | Array |  |

## `_reshape(array: Array, sizes: Array.<number>): Array`

Iteratively re-shape a multi dimensional array to fit the specified dimensions (from math.js)

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| array | Array |  | Array to be reshaped |
| sizes | Array.<number> |  | List of sizes for each dimension |

## `flatten(array: Array): Array`

Flatten a multi dimensional array, put all elements in a one dimensional array

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| array | Array |  | A multi dimensional array |