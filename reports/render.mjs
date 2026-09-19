import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { runInNewContext } from 'node:vm';

const templateUrl = new URL('./report.html', import.meta.url);
const dataPattern = /(<script id="report-data" type="application\/json">)[\s\S]*?(<\/script>)/;

export async function createReport(data) {
  const template = await readFile(templateUrl, 'utf8');
  // Run only the trusted template's pure helpers, never caller-supplied code.
  // Browser and CLI validation share one implementation.
  const core = template.match(/<script id="report-core">([\s\S]*?)<\/script>/)?.[1];
  if (!core || !dataPattern.test(template)) throw new Error('Report template is missing its data or validation block.');
  const validate = runInNewContext(`${core}\nvalidateReport;`, {}, { timeout: 1000 });
  validate(data);
  const json = JSON.stringify(data, null, 2).replace(/</g, '\\u003c');
  return template.replace(dataPattern, (_, open, close) => `${open}\n${json}\n${close}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const args = process.argv.slice(2);
    if (args.length !== 2) throw new Error('Usage: node reports/render.mjs input.json output.html');
    const [input, output] = args;
    const data = JSON.parse(await readFile(input, 'utf8'));
    const html = await createReport(data);
    // Keep prior reports intact. Choose a new output path for each run.
    await writeFile(output, html, { flag: 'wx' });
    console.log(`Created ${output}${data.sample ? ' (illustrative data)' : ''}`);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
