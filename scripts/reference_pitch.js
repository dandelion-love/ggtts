#!/usr/bin/env node
// Run a reference recording-booth pitch calculation on samples supplied over stdin. No audio is written.
// Usage: node reference_pitch.js /path/to/page.js EXPECTED_PAGE_SHA256 < input.json
// page.js is the reference analysis script you supply; it is checked against the hash you give. The detector is the
// Pitchy and fft.js bundled under instruments/vendor, checked against their pinned hashes.
// input.json: {"sample_rate": 24000, "samples": [..]}; stdout contains only aggregate pitch measurements.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const Module = require("module");
const vm = require("vm");

const pageFile = process.argv[2];
const pageSha = process.argv[3];
if (!pageFile || !/^[a-f0-9]{64}$/.test(pageSha || "")) throw Error("reference page.js path and its expected SHA-256 required");
function pinned(file, expected) {
  const bytes = fs.readFileSync(file);
  if (crypto.createHash("sha256").update(bytes).digest("hex") !== expected) throw Error("reference hash mismatch: " + file);
  return bytes.toString("utf8");
}
const vendor = path.join(__dirname, "..", "instruments", "vendor");
const pitchFile = path.join(vendor, "pitchy-4.1.0", "index.js");
const fftFile = path.join(vendor, "fft.js-4.0.4", "fft.js");
const pitchSource = pinned(pitchFile, "79b92333cde9ec16798757f73b624abd13522da31e8fab2993c19d319b3bb36e");
pinned(fftFile, "4092b92cec888377666265b724d676628d377d4d5f56b03b76d9c2c27fd061ea");
const page = pinned(pageFile, pageSha);
const start = page.indexOf("function framesDb(");
const end = page.indexOf("function nudges(", start);
if (start < 0 || end < 0) throw Error("recording analysis boundaries not found");
const mod = new Module(pitchFile);
mod.filename = pitchFile;
mod.paths = Module._nodeModulePaths(path.dirname(pitchFile));
const compiled = pitchSource.replace('import FFT from "fft.js";', "const FFT = require(" + JSON.stringify(fftFile) + ");")
  .replace(/export class /g, "class ") + "\nmodule.exports = { PitchDetector };\n";
if (compiled.includes("export class") || compiled.includes('import FFT from "fft.js";')) throw Error("unsupported reference module");
mod._compile(compiled, pitchFile);
const input = JSON.parse(fs.readFileSync(0, "utf8"));
if (!Number.isInteger(input.sample_rate) || input.sample_rate <= 0 || !Array.isArray(input.samples) ||
    !input.samples.every(Number.isFinite)) throw Error("invalid samples or rate");
const context = { Float32Array, Float64Array, Math, PitchDetector: mod.exports.PitchDetector, S: {roomFloor: null} };
context.globalThis = context;
vm.createContext(context);
vm.runInContext(page.slice(start, end), context, {filename: pageFile});
const result = context.analyse(Float32Array.from(input.samples), input.sample_rate, 0);
const checks = result.checks || {};
console.log(JSON.stringify({pitch_range_st: checks.pitch_range_st == null ? null : checks.pitch_range_st,
                            pitch_median_hz: checks.pitch_median_hz == null ? null : checks.pitch_median_hz,
                            voiced_frames: checks.voiced_frames == null ? 0 : checks.voiced_frames}));
