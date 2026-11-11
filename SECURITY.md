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

## Client-code sessions (`--allow-client-code`)

Sessions let a client ship a code bundle and a dependency lock. This is a
**deliberate change of the trust model**, off by default:

- `session-open` and code bundles are only accepted with
  `--allow-client-code` (or `VRAMD_ALLOW_CLIENT_CODE=1`). With that flag the
  token is equivalent to **remote code execution** on the server;
- bundles are extracted only after every member is checked against the
  client's manifest: relative paths, no `..`, no symlinks/hardlinks or device
  files, per-file and total size caps, and SHA-256 per file. Extraction happens
  into a temporary directory and is published atomically;
- a session backend runs with the daemon's user, filesystem and network. Use a
  dedicated service user, keep the cache on a dedicated volume and put the
  listener behind WireGuard/a trusted LAN;
- `--allow-client-env` additionally lets the client's freeze build a venv with
  `uv` (package build scripts execute). Disable it if the client is not
  trusted; pre-build envs with `vramd env build` instead;
- model downloads (`vramd models ensure`) and sessions write only under
  `HF_HOME` and the vramd cache; the job payload paths are still trusted, so do
  not treat the sandbox as a security boundary between mutually untrusted
  clients. The token is single-tenant.

A request can never supply `runtime.command`/`env` directly; only a session
bundle accepted under these flags can. Job payloads remain data.

## Reporting

Vulnerabilities: open a private [security advisory][adv]. For everything else,
a regular issue.

[adv]: https://github.com/maikramer/vramd/security/advisories/new

Supported versions: the latest `0.x`. While the project is in `0.x`, only the
most recent version receives fixes.
