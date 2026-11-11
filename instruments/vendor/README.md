# Bundled pitch dependencies

`pitchy-4.1.0/index.js` and its MIT licence (`pitchy-4.1.0/LICENSE`) are from the Pitchy 4.1.0 npm package.
`fft.js-4.0.4/fft.js` and `package.json` are from the fft.js 4.0.4 npm package. fft.js ships no licence file: its MIT
licence is the `LICENSE` section of its `README.md`, and `fft.js-4.0.4/LICENSE` is that section, unchanged, from
<https://github.com/indutny/fft.js> at tag `v4.0.4` (commit `f8be92e1369f684da3e121e4c5b7fbcc8d50f868`, the npm
package's `gitHead`). Both licence files must stay beside the sources wherever this directory is copied; the release
archives carry them, and `scripts/package_and_smoke.sh smoke` refuses an archive without them.

The pitch instrument checks the source hashes before executing them; no npm installation or network access is
required. A Node.js runtime is still required.
