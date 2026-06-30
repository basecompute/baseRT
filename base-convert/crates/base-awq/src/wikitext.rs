//! WikiText-103 calibration data loader.
//!
//! WikiText-103 is the default calibration set per the canonical-quant
//! spec: ~100M tokens of cleaned Wikipedia, ~516K lines. The default
//! AWQ paper config samples ~512 sequences × 2048 tokens.
//!
//! On-disk: a plain UTF-8 text file with one wikitext line per text
//! line (matches the standard distribution at
//! `wikitext-103-raw/wiki.train.raw` from the HF Datasets mirror).
//! No Python deps; the user downloads the file once and points the
//! converter at it via `--calib-file <path>`.
//!
//! This module deliberately doesn't tokenize — that's the runtime's
//! job at calibration time, since it must use the model's own
//! tokenizer to be faithful. We just provide line-level access.

use anyhow::{anyhow, Context, Result};
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Iterator over WikiText lines, filtering empty / heading-only
/// (`= Heading =`) lines. The standard preprocessing for AWQ /
/// PPL calibration drops these because they're not representative
/// running text.
pub struct WikiTextReader {
    inner: BufReader<std::fs::File>,
}

impl WikiTextReader {
    /// Open a WikiText file (`wiki.train.raw` or any UTF-8 text file
    /// with one paragraph per line).
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening calibration file {}", path.display()))?;
        Ok(Self {
            inner: BufReader::new(file),
        })
    }

    /// Pull up to `n` non-empty, non-heading lines. Concatenates them
    /// with newline separators — calibration consumers typically tokenize
    /// the joined string.
    pub fn read_n_lines(&mut self, n: usize) -> Result<String> {
        let mut out = String::new();
        let mut count = 0;
        let mut line = String::new();
        while count < n {
            line.clear();
            let read = self
                .inner
                .read_line(&mut line)
                .context("reading WikiText line")?;
            if read == 0 {
                break;
            }
            if is_skippable(&line) {
                continue;
            }
            out.push_str(line.trim_end_matches('\n'));
            out.push('\n');
            count += 1;
        }
        if count == 0 {
            return Err(anyhow!("calibration file has no usable lines"));
        }
        Ok(out)
    }

    /// Pull approximately `target_chars` characters of running text
    /// from the file. Useful when the consumer has a token budget
    /// (typical AWQ paper: ~1M characters ≈ 256K BPE tokens).
    pub fn read_chars(&mut self, target_chars: usize) -> Result<String> {
        let mut out = String::new();
        let mut line = String::new();
        while out.len() < target_chars {
            line.clear();
            let read = self
                .inner
                .read_line(&mut line)
                .context("reading WikiText line")?;
            if read == 0 {
                break;
            }
            if is_skippable(&line) {
                continue;
            }
            out.push_str(line.trim_end_matches('\n'));
            out.push('\n');
        }
        if out.is_empty() {
            return Err(anyhow!("calibration file has no usable text"));
        }
        Ok(out)
    }
}

fn is_skippable(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return true;
    }
    // WikiText section headers: " = Title = ", " = = Subsection = = ".
    if trimmed.starts_with('=') && trimmed.ends_with('=') {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn skips_empty_lines_and_headings() {
        let f = write_temp(
            " \n = Title = \n  \nFirst paragraph.\n \n = = Subsection = = \nSecond.\n",
        );
        let mut r = WikiTextReader::open(f.path()).unwrap();
        let text = r.read_n_lines(5).unwrap();
        assert!(text.contains("First paragraph."));
        assert!(text.contains("Second."));
        assert!(!text.contains("="));
    }

    #[test]
    fn read_n_lines_stops_at_n() {
        let lines: Vec<String> = (0..20).map(|i| format!("line {i}")).collect();
        let f = write_temp(&lines.join("\n"));
        let mut r = WikiTextReader::open(f.path()).unwrap();
        let got = r.read_n_lines(5).unwrap();
        let actual_lines: Vec<&str> = got.lines().collect();
        assert_eq!(actual_lines.len(), 5);
        assert_eq!(actual_lines[0], "line 0");
        assert_eq!(actual_lines[4], "line 4");
    }

    #[test]
    fn read_chars_stops_when_target_reached() {
        let lines: Vec<String> = (0..100).map(|i| format!("line {i:03}")).collect();
        let f = write_temp(&lines.join("\n"));
        let mut r = WikiTextReader::open(f.path()).unwrap();
        let got = r.read_chars(50).unwrap();
        assert!(got.len() >= 50, "got {} chars", got.len());
        assert!(got.len() < 100, "should stop near target, got {}", got.len());
    }

    #[test]
    fn empty_file_errors() {
        let f = write_temp("");
        let mut r = WikiTextReader::open(f.path()).unwrap();
        assert!(r.read_n_lines(1).is_err());
    }

    #[test]
    fn only_headings_errors() {
        let f = write_temp(" = Title = \n = = Sub = = \n");
        let mut r = WikiTextReader::open(f.path()).unwrap();
        assert!(r.read_n_lines(1).is_err());
    }
}
