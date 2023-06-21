const { readFileSync } = require("fs");
const text = readFileSync("./vocab.json");
const vocab = JSON.parse(text);

console.log(Object.keys(vocab).length);
