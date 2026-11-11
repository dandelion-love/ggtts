#!/usr/bin/env node
// Run the bundled Pitchy detector on mono float32 samples from stdin. The statistic and gating mirror the reference booth.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const Module = require("module");

const vendor = path.resolve(__dirname, "../../../vendor");
const pitchPath = path.join(vendor, "pitchy-4.1.0/index.js");
const fftPath = path.join(vendor, "fft.js-4.0.4/fft.js");
function verified(file, expected) {
  const data = fs.readFileSync(file);
  if (crypto.createHash("sha256").update(data).digest("hex") !== expected) throw Error("pitch dependency hash mismatch");
  return data.toString("utf8");
}
const source = verified(pitchPath, "79b92333cde9ec16798757f73b624abd13522da31e8fab2993c19d319b3bb36e");
verified(fftPath, "4092b92cec888377666265b724d676628d377d4d5f56b03b76d9c2c27fd061ea");
const mod = new Module(pitchPath);
mod.filename = pitchPath;
mod.paths = Module._nodeModulePaths(path.dirname(pitchPath));
const compiled = source.replace('import FFT from "fft.js";', "const FFT = require(" + JSON.stringify(fftPath) + ");")
  .replace(/export class /g, "class ") + "\nmodule.exports = { PitchDetector };\n";
if (compiled.includes('import FFT from "fft.js";') || compiled.includes("export class")) throw Error("unsupported pitch dependency");
mod._compile(compiled, pitchPath);

const input = fs.readFileSync(0);
if (input.length < 4 || (input.length - 4) % 4) throw Error("invalid float32 audio stream");
const rate = input.readUInt32LE(0);
if (!rate) throw Error("invalid sample rate");
const samples = new Float32Array((input.length - 4) / 4);
for (let i = 0; i < samples.length; i++) {
  const x = input.readFloatLE(4 + 4 * i);
  if (!Number.isFinite(x)) throw Error("non-finite sample");
  samples[i] = x;
}
const hop = Math.round(rate * 0.01);
if (!hop) throw Error("sample rate too low");
const db = [];
for (let start = 0; start + hop <= samples.length; start += hop) {
  let sum = 0;
  for (let j = start; j < start + hop; j++) sum += samples[j] * samples[j];
  db.push(20 * Math.log10(Math.sqrt(sum / hop) + 1e-9));
}
function percentile(values, p) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p / 100 * (sorted.length - 1))))];
}
let voiced = [];
const win = rate >= 44100 ? 2048 : 1024;
if (db.length && samples.length >= win) {
  let max = -200;
  for (const value of db) if (value > max) max = value;
  const threshold = Math.max(percentile(db, 10) + 10, max - 35);
  const detector = mod.exports.PitchDetector.forFloat32Array(win);
  for (let f = 0; f * hop + win <= samples.length; f++) {
    const centre = db[f + Math.floor(win / hop / 2)];
    if (centre === undefined || centre < threshold) continue;
    const [hz, clarity] = detector.findPitch(samples.subarray(f * hop, f * hop + win), rate);
    if (clarity > 0.9 && hz > 60 && hz < 500) voiced.push(hz);
  }
}
let median = null, range = null;
if (voiced.length >= 10) {
  median = percentile(voiced, 50);
  const semitones = voiced.map(hz => 12 * Math.log2(hz / median));
  range = percentile(semitones, 95) - percentile(semitones, 5);
}
console.log(JSON.stringify({voiced_frames: voiced.length, pitch_median_hz: median, pitch_range_st: range}));
