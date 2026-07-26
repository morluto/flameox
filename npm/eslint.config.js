"use strict";

const js = require("@eslint/js");

module.exports = [
  js.configs.recommended,
  {
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "commonjs",
      globals: {
        process: "readonly",
        require: "readonly",
        module: "readonly",
        __dirname: "readonly",
        setTimeout: "readonly",
        Buffer: "readonly",
        console: "readonly",
      },
    },
    rules: {
      "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "no-undef": "error",
      complexity: ["warn", 10],
      "max-lines": ["warn", { max: 200, skipComments: true, skipBlankLines: true }],
      "max-depth": ["warn", 4],
      "no-console": "off",
    },
  },
  {
    files: ["test/**/*.test.cjs"],
    languageOptions: {
      globals: {
        process: "readonly",
        require: "readonly",
        module: "readonly",
        __dirname: "readonly",
        setTimeout: "readonly",
        Buffer: "readonly",
        console: "readonly",
      },
    },
  },
];
