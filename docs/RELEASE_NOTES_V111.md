# v1.1.1 — unpublished CI candidate

Version 1.1.1 was a tagged but unreleased CI candidate. It proved that the
frozen manifest bytes were portable, then exposed a Linux-specific direct-script
import failure in the workflow. It has no Zenodo version and no GitHub release;
v1.1.2 supersedes it. No scientific result, prediction, label, analysis,
theorem, figure, or reported number changed.

The patch marks the prospectively frozen aggregate prediction-lock manifest as
a binary Git object. This preserves its original CRLF bytes across Windows and
Linux checkouts, so the historical SHA-256 recorded before target-label
opening verifies identically in GitHub Actions. The v1.1.0 release remains an
immutable predecessor documenting the initially published snapshot.

The final patch additionally invokes the manuscript validator as a module in
GitHub Actions, which preserves the repository root on `sys.path` under both
supported Python versions.
