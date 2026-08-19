# Superseded F3-F1S Session

This session was superseded before any test access because the prepared F3-F2
reporting runner used the wrong curve identifier column during table merging.
The frozen runner hash therefore no longer matches this session's protocol,
and F3-F2 will refuse to use it.

Use `../20260818T211629` instead. No test features or targets were read by
either F3-F1S session.
