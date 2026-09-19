import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';
import { createReport } from '../reports/render.mjs';

const html = await readFile(new URL('../reports/report.html', import.meta.url), 'utf8');
const core = html.match(/<script id="report-core">([\s\S]*?)<\/script>/)[1];
const { validateReport, formatMetric, changeLabel, axisScale } = runInNewContext(
  `${core}\n({ validateReport, formatMetric, changeLabel, axisScale });`,
);
const sample = () => JSON.parse(html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);

test('report specimen stays explicitly illustrative and standalone', () => {
  const data = validateReport(sample());
  assert.equal(data.sample, true);
  assert.equal(data.iterations.length, 6);
  assert.equal((html.match(/data:font\/woff2;base64,/g) ?? []).length, 2);
  assert.doesNotMatch(html, /<(?:script|link|img)[^>]+(?:src|href)=["']https?:/);
});

test('missing measurements stay distinct from zero; memory uses binary GiB', () => {
  assert.equal(formatMetric(null, 'outputTokensPerSecond'), 'N/A');
  assert.equal(formatMetric(undefined, 'modelMemoryBytes'), 'N/A');
  assert.equal(formatMetric(0, 'ttftMs'), '0');
  assert.equal(formatMetric(1073741824, 'modelMemoryBytes'), '1.00');
  const data = sample();
  data.iterations = [{ id: '001', outputTokensPerSecond: null, ttftMs: 0 }];
  assert.doesNotThrow(() => validateReport(data));
});

test('comparison language preserves direction and avoids undefined percentages', () => {
  assert.equal(changeLabel(75, 100, 'outputTokensPerSecond'), '25.0% slower');
  assert.equal(changeLabel(75, 100, 'ttftMs'), '25.0% lower latency');
  assert.equal(changeLabel(125, 100, 'modelMemoryBytes'), '25.0% more memory');
  assert.equal(changeLabel(100, 100, 'ttftMs'), 'Unchanged');
  assert.equal(changeLabel(100, 0, 'ttftMs'), null);
  assert.equal(changeLabel(null, 100, 'ttftMs'), null);
  assert.equal(changeLabel(100, null, 'ttftMs'), null);
});

test('invalid metrics and inconsistent run configuration are rejected', () => {
  const edits = [
    d => { d.iterations[0].ttftMs = -1; },
    d => { d.iterations[0].outputTokensPerSecond = '100'; },
    d => { d.iterations[0].prefillTokensPerSecond = Infinity; },
    d => { d.iterations[0].peakMemoryBytes = 1; },
    d => { d.iterations[1].id = d.iterations[0].id; },
    d => { d.io.batchSize = 0; },
    d => { d.io.inputTokens = d.io.contextTokens + 1; },
    d => { d.measurement.repetitions = 2.5; },
    d => { d.iterations = []; },
    d => { delete d.sample; },
  ];
  for (const edit of edits) {
    const data = sample();
    edit(data);
    assert.throws(() => validateReport(data));
  }
});

test('zero and fractional chart series have a finite nonzero scale', () => {
  for (const values of [[0, 0], [0.001, 0.002], [0.94, 1.03], [124.8, 186.4], [1000, 3000]]) {
    const scale = axisScale(values);
    assert.ok(Number.isFinite(scale.max) && scale.max > 0);
    assert.ok(Number.isFinite(scale.step) && scale.step > 0);
    assert.ok(scale.max >= Math.max(...values));
  }
});

test('generated reports escape script delimiters and preserve arbitrary text', async () => {
  const data = sample();
  data.model.name = '</script><script>alert("x")</script> $&';
  data.sample = false;
  const result = await createReport(data);
  const embedded = result.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  assert.deepEqual(JSON.parse(embedded), data);
  assert.doesNotMatch(embedded, /</);
  assert.equal((result.match(/<script /g) ?? []).length, 3);
});

test('report generation rejects invalid input before writing an artifact', async () => {
  const data = sample();
  data.schemaVersion = 999;
  await assert.rejects(createReport(data), /schemaVersion/);
});
