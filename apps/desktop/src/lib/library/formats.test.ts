import assert from 'node:assert/strict'
import { test } from 'node:test'

import { formatLine, limitLine, limitPhrase, sizeLabel } from './formats.ts'

test('the formats line names every served extension once, in the served order', () => {
  assert.equal(
    formatLine(['.pdf', '.epub', '.html', '.htm', '.docx', '.odt', '.rtf', '.md', '.txt']),
    'PDF · EPUB · HTML · DOCX · ODT · RTF · MD · TXT',
  )
})

test('.htm is folded into HTML only when .html is also served', () => {
  assert.equal(formatLine(['.htm']), 'HTM')
  assert.equal(formatLine(['.html', '.htm']), 'HTML')
})

test('the default ingest cap reads as 1 GB, not 1024 MB or 1.0 GB', () => {
  assert.equal(sizeLabel(1073741824), '1 GB')
  assert.equal(limitLine({ max_file_bytes: 1073741824 }), 'Up to 1 GB per file')
})

test('the mid-sentence phrase keeps the unit upper-case (caught live: "1 gb")', () => {
  assert.equal(limitPhrase({ max_file_bytes: 1073741824 }), 'up to 1 GB per file')
})

test('a configured cap that is not a whole unit keeps one decimal', () => {
  assert.equal(sizeLabel(1.5 * 1024 ** 3), '1.5 GB')
  assert.equal(sizeLabel(500 * 1024 ** 2), '500 MB')
  assert.equal(sizeLabel(1000), '1000 bytes')
})
