# Class

## `BaseNeuralNetwork`

Deep Learning with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `model: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

### `train(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>): Object`

Asynchronously trains tensorflow model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | array of model dense layer parameters |

### `calculate(matrix: Array<Array<number>>|Array<number>): {data: Promise}`

Predicts new dependent variables

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| matrix | Array<Array<number>>|Array<number> |  | new test independent variables |

## `DeepLearningRegression`

Deep Learning Regression with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object,layerPreference:String}, properties: {model:Object,tf:Object,})`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow regression model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

## `DeepLearningClassification`

Deep Learning Classification with Tensorflow

### `constructor(options: {layers:Array<Object>,compile:Object,fit:Object}, properties: {model:Object,tf:Object,})`

### `yShape: *`

### `xShape: *`

### `generateLayers(x_matrix: Array<Array<number>>, y_matrix: Array<Array<number>>, layers: Array<Object>)`

Adds dense layers to tensorflow classification model

| Name | Type | Attribute | Description |
| --- | --- | --- | --- |
| x_matrix | Array<Array<number>> |  | independent variables |
| y_matrix | Array<Array<number>> |  | dependent variables |
| layers | Array<Object> |  | model dense layer parameters |

# Function