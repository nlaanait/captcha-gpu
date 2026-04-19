# `captcha-gpu`

This monorepo contains tools for monitoring, managing, and forecasting GPU instance availability on Lambda Cloud.

## Projects

### 1. `lambda-cloud-api`

A command-line client to monitor and manage Lambda Cloud resources.

- **Monitor Availability**: Watch for specific GPU types (A100, GH200, etc.) and get alerts.
- **Launch & Manage**: Interactively launch instances and manage firewall rules.
- **Pricing & Status**: View current instances, regions, and pricing.
- **Track Availability**: Sample the availability of various instances across regions.

### 2. `gpu-forecaster`

A machine learning package for forecasting GPU availability trends using historical data.

- **Availability Prediction**: Predicts future availability of high-demand GPU types.
- **NeuralProphet Models**: Utilizes NeuralProphet for time-series forecasting.
- **Data Analysis**: Analyzes historical availability patterns.

## Setup

1. **Get your API key**: [Lambda Cloud API Keys](https://cloud.lambdalabs.com/api-keys).
2. **Environment**: Copy `env.example` to `.env` and set your `LAMBDA_CLOUD_API_KEY`.
3. **Dependencies**:
   - **lambda-cloud-api**: Requires [Rust/Cargo](https://rustup.rs).
   - **gpu-forecaster**: Requires [Pixi](https://pixi.sh).

## Usage

Most monorepo commands can be run from the root directory using [Pixi](https://pixi.sh).

### Monorepo API Commands (Proxied to Rust tool)

```bash
pixi run monitor        # Watch for specific instance types
pixi run stats          # Sample instances availability timeline and trigger retraining
pixi run instances      # List your launched instances
pixi run check          # One-time check for specific GPU availability
pixi run types          # List all instance types and their status
pixi run captcha-gpu    # Forecast-driven polling strategy for acquiring GPUs
```

### Forecasting & Modeling Tasks

```bash
pixi run train          # Train NeuralProphet models on historical data
pixi run predict        # Run manual prediction for a specific GPU/time
pixi run strategy       # Generate detailed polling strategy JSON
pixi run evaluate       # Evaluate model performance across all GPUs
```

### Direct Rust Tool Usage

Alternatively, you can run the API client directly from its directory:

```bash
cd lambda-cloud-api
cargo run --release -- [COMMAND]
```

## Security

- API keys are loaded from environment variables or a local `.env` file.
- Uses Basic Auth as required by the Lambda Cloud API.
