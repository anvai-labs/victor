# Self-hosted runner environments

Victor's default Linux jobs use the explicit GitHub-hosted `ubuntu-24.04`
image. Under Victor's runner-label policy, self-hosted capacity is opt-in: a job
must request `self-hosted` and the capability labels that describe its workload.
This keeps an older host, a different CPU, or a contaminated tool cache from
silently changing the test environment.

## Why Docker does not remove host differences

Installing Docker on a runner only makes the Docker daemon available. GitHub
Actions still executes checkout, Python setup, package installation and tests
on the host unless the job declares a `container:` or a step explicitly uses a
container action. Victor uses containers for image builds and scanners; its
ordinary test jobs are host jobs.

Even a job-level container shares the host kernel and CPU instruction set. It
can standardize user-space libraries such as glibc, but it cannot make an
unsupported native wheel safe on an older CPU. Native Python, PyO3 and LanceDB
jobs therefore need both an OS/ABI contract and a compatible CPU class.

## Required baseline

A general Victor Linux runner must provide:

- Ubuntu 24.04 with current security updates and glibc 2.38 or newer;
- an x86-64 CPU compatible with every native wheel assigned to the runner;
- enough free disk for the checked-out tree, package caches and build outputs;
- Docker only when the runner advertises the `docker-host` capability;
- an ephemeral or routinely cleaned workspace and tool cache;
- a current GitHub Actions runner service with no repository secrets baked into
  its image.

Run the repository preflight from a clean checkout before registering labels:

```bash
python3 scripts/ci/runner_preflight.py \
  --require-os ubuntu \
  --min-os-version 24.04 \
  --min-glibc 2.38
```

Add `--require-command docker` for a Docker-capable pool. Then exercise the
actual Python 3.12 setup and native imports assigned to that pool; a glibc check
cannot detect an illegal CPU instruction in a third-party wheel.

## Upgrade and reimage procedure

Do not copy `/lib`, replace the dynamic loader, or install a foreign `libc6`
package in place. glibc is part of the distribution ABI; a partial replacement
can make the shell, package manager and runner service unstartable. Reimage an
older runner with Ubuntu 24.04, or perform a controlled distribution upgrade
from its console and reboot it before returning it to service.

For a runner already on Ubuntu 24.04, an operator with passwordless sudo can
apply normal distribution updates:

```bash
sudo -n apt-get update
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -y dist-upgrade
sudo -n reboot
```

After reconnecting, verify the result and clear any tool cache populated by a
different OS image:

```bash
cat /etc/os-release
getconf GNU_LIBC_VERSION
python3 scripts/ci/runner_preflight.py \
  --require-os ubuntu \
  --min-os-version 24.04 \
  --min-glibc 2.38
```

For WSL runners, update or replace the WSL distribution rather than transplanting
glibc. Stop the Actions service before the upgrade and start it only after the
preflight passes.

## Labels and routing

Do not give a self-hosted machine GitHub-hosted image labels such as
`ubuntu-latest` or `ubuntu-24.04`. Use labels that express verified capabilities,
for example:

```yaml
runs-on: [self-hosted, Linux, X64, victor-ci, noble, ci-general]
```

Native jobs should add a separately verified CPU label. Docker jobs should add
`docker-host`. Remove a capability label before maintenance or when a preflight
fails; do not let an incompatible machine remain eligible while it is repaired.

The workflow policy tests reject `ubuntu-latest` in Victor workflows and protect
the jobs whose Python/native ABI requirements previously exposed old hosts.
