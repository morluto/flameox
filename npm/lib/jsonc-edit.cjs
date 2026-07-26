#!/usr/bin/env node

"use strict";

const {
  applyEdits,
  modify,
  parse,
  printParseErrorCode,
} = require("jsonc-parser");

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  try {
    const request = JSON.parse(input);
    if (request.operation === "parse") {
      const errors = [];
      const value = parse(request.text, errors, {
        allowTrailingComma: true,
        disallowComments: false,
      });
      if (errors.length > 0) {
        const detail = errors
          .map((error) => `${printParseErrorCode(error.error)} at offset ${error.offset}`)
          .join(", ");
        throw new Error(detail);
      }
      process.stdout.write(JSON.stringify({ value }));
      return;
    }
    if (request.operation === "modify") {
      const value = request.remove ? undefined : request.value;
      const edits = modify(request.text, request.path, value, {
        formattingOptions: {
          insertSpaces: true,
          tabSize: 2,
          eol: request.text.includes("\r\n") ? "\r\n" : "\n",
        },
      });
      process.stdout.write(JSON.stringify({ text: applyEdits(request.text, edits) }));
      return;
    }
    throw new Error("unsupported JSONC operation");
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 1;
  }
});
