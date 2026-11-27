// Wire contract for the library summary — mirrors apps/api/models/summary.py (ROADMAP 112).
// Counts only, never a ratio: a collection of 0 documents has nothing to divide by.

/** Every issue *Show the N* can open; `library.summary.ISSUES` defines each one, once. */
export type SummaryIssue =
  | 'no_text'
  | 'flagged'
  | 'missing_title'
  | 'missing_authors'
  | 'missing_year'
  | 'missing_doi'
  | 'no_references'
  | 'no_figures'
  | 'no_keywords'
  | 'letter_spaced'
  | 'addressed_to_ai'
  | 'text_not_checked'
  | 'references_heading_only'
  | 'references_none'
  | 'references_few'
  | 'references_not_looked_for'
  | 'figures_uncaptioned'
  | 'tables_missed'

export interface SummaryIndexing {
  no_text: number
  flagged_marginal: number
  flagged_broken: number
  not_classified: number
  /** `null` in a folder's scope: files that are not documents belong to no folder. */
  files_not_in_library: number | null
  files_excluded: number | null
  /** The documents folder's last look (a scan); `null` when it was never looked at. */
  files_seen_at: string | null
  /** Failures in this session's last index run; `null` when there was no run this session. */
  last_run_failed: number | null
}

export interface SummaryMetadata {
  missing_title: number
  missing_authors: number
  missing_year: number
  missing_doi: number
  implausible_year: number
  set_by_you: number
  from_a_catalogue: number
  /** Documents with any of title, authors, year, DOI. 0 with documents present: nothing has read
   *  metadata yet (indexing reads none), which the screen says once, not as four rows. */
  with_any_field: number
}

/** Mirrors apps/api/models/summary.py::RecordCountsPayload. */
export interface SummaryRecord {
  recorded: number
  /** A reference list on record: the record's found/partial, or rows from a batch run. */
  references_listed: number
  references_heading_only: number
  references_none: number
  references_few: number
  references_not_looked_for: number
  figures_uncaptioned_documents: number
  figures_uncaptioned: number
  tables_extracted: number
  tables_documents: number
  tables_missed: number
  tables_not_looked_for: number
}

export interface SummaryOnRecord {
  on_record: number
  none_on_record: number
  rows: number
}

export interface SummaryFigures {
  on_record: number
  none_on_record: number
  rows: number
  described: number
  pdfs: number
}

export interface CollectionSummary {
  scope: 'library' | 'folder'
  folder_id: string | null
  folder_name: string | null
  counted_at: string
  documents: number
  passages: number
  pages: number
  paged_documents: number
  need_attention: number
  indexing: SummaryIndexing
  metadata: SummaryMetadata
  found: {
    references: SummaryOnRecord
    figures: SummaryFigures
    letter_spacing: { recorded: number; pdfs: number }
    keywords: SummaryOnRecord
    /** A2b: documents a current check marked, documents checked, and documents with text and no
     *  current check (never checked, or their text or the patterns changed since). */
    text_checks: { marked: number; checked: number; not_checked: number }
    /** 0.9.0: what the newest extraction records say (`recorded` documents have one). */
    record: SummaryRecord
  }
  types: { format: string; documents: number }[]
  years: {
    by_decade: { decade: number; documents: number }[]
    no_year: number
    oldest: number | null
    newest: number | null
  }
  folders: { in_a_folder: number | null; in_none: number | null; also_elsewhere: number | null }
  added: { first: string | null; last: string | null }
}
