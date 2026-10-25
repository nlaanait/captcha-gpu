---
name: captcha-gpu
description: >-
  Operate the captcha-gpu CLI to check, monitor, and forecast Lambda Cloud GPU
  availability and launch instances. Use when the user mentions captcha-gpu,
  Lambda Cloud, GH200, A100, A10, GPU capacity, or asks what is available right
  now, to watch for capacity, or to acquire an instance.
---

# captcha-gpu

`captcha-gpu` checks, watches, and acquires Lambda Cloud GPU instances. Launching bills the account immediately. Do not launch, and do not pass `--auto-launch`, unless the user asked to acquire an instance.

## Run

Prefer `captcha-gpu` on `PATH`. From the repo root, the fallback is:

```bash
cargo run --manifest-path lambda-cloud-api/Cargo.toml --bin captcha-gpu --release -- <mode>
```

`LAMBDA_CLOUD_API_KEY` is read from the process environment first. If it is unset, it is filled from `$CAPTCHA_GPU_ROOT/.env` (the global `captcha-gpu` command sets that to the repo) and then from `.env` in the working directory or its parents. If it is missing, ask the user. Do not invent a key and do not print it.

The same order applies to `LAMBDA_SSH_KEY` (otherwise the first key on the account), `MY_PUBLIC_IP` (restricts firewall ports 22/80/443), and `LAMBDA_INSTANCE_PREFIX` (default `lambda-captcha`).

## Modes

**check** — one request, no launch. Use for "what is available right now".

```bash
captcha-gpu check --types gh200
```

`--types` is a comma-separated list (`gh200`, `a100`, `a10`). Omitted, it checks a10, a100, and gh200.

**monitor** — poll until stopped. Use when the user wants to wait for capacity.

```bash
captcha-gpu monitor --types gh200 --interval 30
```

Default interval is 30 seconds. It asks before launching and asks which region. `--auto-launch` skips both prompts, launches the first matching type in the first region with capacity, and exits. Leave it off unless the user explicitly asked to launch without prompting.

Run monitor in the background. Do not stop a monitor the user already started unless they ask.

**forecast** — predict the best hour, then poll inside that window to acquire a GPU.

```bash
captcha-gpu forecast --gpu gpu_1x_gh200
```

`--gpu` is the full instance type and must match a model in `gpu-forecaster/models/` (`gpu_1x_gh200`, `gpu_1x_a10`, `gpu_1x_a100_sxm4`, `gpu_8x_a100`, `gpu_8x_a100_80gb_sxm4`). `--top` defaults to 3. `--start` is `YYYY-MM-DD HH:MM:SS`. `--dry-run` polls but does not launch.

Forecast prompts for a timeslot and needs a TTY. For an unattended launch, use `monitor --auto-launch` instead.

## Which mode

- Available right now → `check`
- Watch until capacity appears → `monitor` without `--auto-launch`
- Acquire as soon as capacity exists → `monitor --types <gpu> --auto-launch`
- Acquire during the forecasted hour → `forecast --gpu <full type>`

`check` and `monitor --types` take short names (`gh200`). `forecast --gpu` takes the full type (`gpu_1x_gh200`).

## Other subcommands

- `captcha-gpu instances` — instances already launched
- `captcha-gpu types` — every instance type and whether it has capacity
- `captcha-gpu stats` — append availability samples to a CSV; long-running

## After a launch

Report the instance id, type, region, public IP, and the SSH config block the tool prints. Do not terminate the instance unless the user asks.
