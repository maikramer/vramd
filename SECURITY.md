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
authentication layer in front.

## Remote mode (`--listen`)

`vramd start --listen HOST:PORT` serves the GPUs over TCP. In this mode:

- a **token is mandatory** (from `--token-file`/`VRAMD_TOKEN_FILE`, or generated
  per start) and is published in the descriptor with mode `0600`;
- the token is a **full-control credential**: it authorises jobs, `zero`,
  `respawn`, `flush` and `shutdown`. There are no per-command permissions;
- the token travels in clear text on every request. Bind to a trusted LAN or
  put the daemon behind WireGuard/SSH; do not expose the port to the Internet;
- file transfer (`stage`/`put`/`get`/`list`/`clean`) is confined to
  `<VRAMD_REMOTE_ROOT>/<job>/` with canonical path validation, per-file and
  per-job quotas and a TTL janitor. The job **payload** paths (`output`, input
  fields) are not interpreted by the supervisor — the worker runs with the
  daemon's privileges, so a client that can submit jobs can already write
  anywhere that process can. Treat remote clients as trusted users;
- the job WAL persists the full request (prompts and paths) next to the
  descriptor. Set `VRAMD_WAL=0` on a shared machine.

The same rules apply to `runtime.command`/`env`: they are **administrator**
configuration and must never come from a request.

## Reporting

Vulnerabilities: open a private [security advisory][adv]. For everything else,
a regular issue.

[adv]: https://github.com/maikramer/vramd/security/advisories/new

Supported versions: the latest `0.x`. While the project is in `0.x`, only the
most recent version receives fixes.
