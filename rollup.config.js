import resolve from 'rollup-plugin-node-resolve';
import commonjs from 'rollup-plugin-commonjs';
// import alias from 'rollup-plugin-alias';
import babel from 'rollup-plugin-babel';
import replace from 'rollup-plugin-replace';
// import async from 'rollup-plugin-async';
import builtins from 'rollup-plugin-node-builtins';
import globals from 'rollup-plugin-node-globals';
import pkg from './package.json';
import { terser, } from 'rollup-plugin-terser';

const plugins = [
  replace({
    'process.env.NODE_ENV': JSON.stringify('development'),
  }),
  resolve({
    preferBuiltins: true,
  }), // so Rollup can find `ms`
  builtins({
  }),
  commonjs({
    namedExports: {
      'node_modules/lodash.range/index.js': [ 'default', ],
      'node_modules/@tensorflow/tfjs/dist/tf.esm.js': [ 'default' ],
    },
  }), // so Rollup can convert `ms` to an ES module
  babel({
    runtimeHelpers: true,
    externalHelpers: true,
    // exclude: 'node_modules/@babel/runtime/**',
    exclude: 'node_modules/@babel/runtime/helpers/typeof.js',
    'presets': [
      [ '@babel/env', {}, ],
    ],
    plugins: [
      [
        '@babel/transform-runtime',
        // { useESModules: output.format !== 'cjs' }
      ],
      [ 'babel-plugin-replace-imports', {
        'test': /tensorflow\/tfjs-node$/,
        'replacer': 'tensorflow/tfjs',
      }, ],
      [
        '@babel/plugin-proposal-export-namespace-from',
      ],
    ],
    // exclude: 'node_modules/**', // only transpile our source code
  }),
  
  globals({
  }),
  
];

const minifiedPlugins = plugins.concat([
  terser({
    sourcemap: true
  }),
]);

export default [
  // browser-friendly UMD build
  {
    input: 'index.js',
    output: [
      {
        exports: 'named',
        file: pkg.browser,
        name: 'tensorscript',
        format: 'umd',
      },
      {
        exports: 'named',
        file: pkg.web,
        name: 'tensorscript',
        format: 'iife',
      },
    ],
    plugins,
  },
  {
    input: 'index.js',
    output: [
      {
        exports: 'named',
        file: pkg.browser.replace('.js','.min.js'),
        name: 'tensorscript',
        format: 'umd',
      },
      {
        exports: 'named',
        file: pkg.web.replace('.js','.min.js'),
        name: 'tensorscript',
        format: 'iife',
      },
    ],
    plugins:minifiedPlugins,
  },

  // CommonJS (for Node) and ES module (for bundlers) build.
  // (We could have three entries in the configuration array
  // instead of two, but it's quicker to generate multiple
  // builds from a single configuration where possible, using
  // an array for the `output` option, where we can specify 
  // `file` and `format` for each target)
  {
    input: 'index.js',
    external: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-node',
      'lodash.range',
      // 'lodash.rangeright'
    ],
    output: [
      {
        exports: 'named',
        file: pkg.node,
        name: 'tensorscript',
        format: 'cjs',
      },
      {
        exports: 'named',
        file: pkg.es,
        name: 'tensorscript',
        format: 'es',
      },
    ],
  },
];
