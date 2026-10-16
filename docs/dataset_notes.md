# Dataset Notes

## AX-b / Broadcoverage Diagnostics

SuperGLUE AX-b is a binary entailment diagnostic set. It is reported alongside
SuperGLUE but is not included in the main aggregate score. The released split is
test-only and has 1104 examples.

## GLUE Diagnostic AX

The original GLUE diagnostic set is also NLI-style, but uses three labels:
`entailment`, `neutral`, and `contradiction`. Its extra linguistic phenomenon
tags are useful for probing whether a graph module helps on negation,
quantifiers, lexical semantics, predicate-argument structure, and similar
categories.

## XNLI

XNLI is the main multilingual comparison set for NLI. It extends MultiNLI dev
and test pairs to 15 languages: English, French, Spanish, German, Greek,
Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili,
and Urdu.

The official XNLI package provides dev/test evaluation data. The standard
training setup is English MultiNLI; translate-train experiments use the optional
XNLI-MT archive.
