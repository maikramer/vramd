# Security

## Threat model

`vramd` is designed to run **locally**, for a single user:

- the socket is a Unix socket with the user's permissions — **no
  authentication**;
- any process of that user can submit jobs, load backends and stop the
  supervisor;
- a backend descriptor defines **commands that will be executed**
  (`runtime.command`). Treat a third-party `backends.yaml` with the same care
  you'd give a script.

Don't expose the socket to the network or share it between users without an
authentication layer in front. An HTTP gateway would require rethinking this
from scratch.

## Reporting

Vulnerabilities: open a private [security advisory][adv]. For everything else,
a regular issue.

[adv]: https://github.com/maikramer/vramd/security/advisories/new

Supported versions: the latest `0.x`. While the project is in `0.x`, only the
most recent version receives fixes.
