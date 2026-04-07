use anyhow::{Context, Result};
use chrono::Local;
use clap::{Parser, Subcommand};
use csv::WriterBuilder;
use dialoguer::{Confirm, Select};
use dotenvy::dotenv;
use lambda_cloud_api::{LambdaCloudClient, LambdaConfig, launch_and_setup};
use rand::Rng;
use serde::Deserialize;
use std::env;
use std::fs::OpenOptions;
use std::path::Path;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Deserialize, Debug)]
struct BlockStrategy {
    block_start: String,
    block_end: String,
    budget_percent: f64,
}

#[derive(Deserialize, Debug)]
struct SlotStrategy {
    timeslot_start: String,
    timeslot_end: String,
    availability_score: f64,
    strategy: String,
    blocks: Vec<BlockStrategy>,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Captcha-GPU: Polling strategy to acquire GPU instance
    CaptchaGpu {
        /// Instance type to capture (e.g., gpu_1x_a10)
        #[arg(short, long)]
        gpu: String,
        /// Start time for the 24h forecast window (YYYY-MM-DD HH:MM:SS). Defaults to current local time.
        #[arg(short, long)]
        start: Option<String>,
        /// Number of top timeslots to return
        #[arg(long, default_value_t = 3)]
        top: usize,
        /// Simulate the launch without actually creating the instance
        #[arg(long)]
        dry_run: bool,
    },
    /// Continuously monitor availability and launch instances
    Monitor {
        /// Instance types to monitor (e.g., a10, a100, gh200)
        #[arg(short, long, value_delimiter = ',')]
        types: Option<Vec<String>>,
        /// Interval in seconds between checks
        #[arg(short, long, default_value_t = 30)]
        interval: u64,
    },
    /// Check availability of specific instance types (one-time)
    Check {
        /// Instance types to check (e.g., a10, a100, gh200)
        #[arg(short, long, value_delimiter = ',')]
        types: Option<Vec<String>>,
    },
    /// List all your launched instances
    Instances,
    /// List all available instance types and their status
    Types,
    /// Collect availability statistics over time
    Stats {
        /// Instance types to monitor (e.g., a10, a100, gh200)
        #[arg(short, long, value_delimiter = ',')]
        types: Option<Vec<String>>,
        /// Minimum interval in seconds between samples
        #[arg(long, default_value_t = 30)]
        min_interval: u64,
        /// Maximum interval in seconds between samples
        #[arg(long, default_value_t = 120)]
        max_interval: u64,
        /// Output file for statistics
        #[arg(short, long, default_value = "availability_stats.csv")]
        output: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let cli = Cli::parse();

    let api_key = env::var("LAMBDA_CLOUD_API_KEY").context("LAMBDA_CLOUD_API_KEY environment variable not set")?;
    let client = LambdaCloudClient::new(api_key);

    match cli.command.unwrap_or(Commands::Instances) {
        Commands::CaptchaGpu { gpu, start, top, dry_run } => {
            let (general_limit, _launch_limit) = LambdaCloudClient::get_api_rate_limit();
            // Respect the 1 req/sec limit (3600/hour) with a safety margin
            let max_hourly_budget = std::cmp::min(3000, (general_limit as f64 * 0.85) as u64);

            let current_time = start.unwrap_or_else(|| Local::now().format("%Y-%m-%d %H:%M:%S").to_string());
            println!("🧠 Asking gpu-forecaster for the best {} timeslots for {} (Start: {})...", top, gpu, current_time);
            
            let top_str = top.to_string();
            let mut command = std::process::Command::new("pixi");
            command.args(["run", "python", "src/api_strategy.py", "--gpu", &gpu, "--start", &current_time, "--top", &top_str, "--json"]);
            
            if Path::new("../gpu-forecaster").exists() {
                command.current_dir("../gpu-forecaster");
            } else if Path::new("gpu-forecaster").exists() {
                command.current_dir("gpu-forecaster");
            } else {
                eprintln!("❌ gpu-forecaster directory not found.");
                return Ok(());
            }
            
            let output = command.output().context("Failed to execute gpu-forecaster")?;

            if !output.status.success() {
                eprintln!("❌ gpu-forecaster failed:");
                eprintln!("{}", String::from_utf8_lossy(&output.stderr));
                return Ok(());
            }

            let stdout_str = String::from_utf8_lossy(&output.stdout);
            
            let json_start = stdout_str.find("[\n  {").unwrap_or_else(|| stdout_str.find('[').unwrap_or(0));
            let json_str = &stdout_str[json_start..];
            
            let strategies: Vec<SlotStrategy> = match serde_json::from_str(json_str) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("❌ Failed to parse JSON from gpu-forecaster: {}", e);
                    eprintln!("Output was: {}", json_str);
                    return Ok(());
                }
            };

            if strategies.is_empty() {
                println!("No timeslots returned.");
                return Ok(());
            }

            let options: Vec<String> = strategies.iter().map(|s| {
                format!("{} to {} (Score: {:.2}%, Strategy: {})", 
                    s.timeslot_start, s.timeslot_end, s.availability_score, s.strategy)
            }).collect();

            let selection = Select::new()
                .with_prompt("Select a timeslot to run the Captcha-GPU polling strategy")
                .items(&options)
                .default(0)
                .interact()?;

            let selected_strategy = &strategies[selection];
            println!("\n✅ Selected timeslot: {} to {}", selected_strategy.timeslot_start, selected_strategy.timeslot_end);
            
            use chrono::NaiveDateTime;
            let start_time = NaiveDateTime::parse_from_str(&selected_strategy.timeslot_start, "%Y-%m-%d %H:%M:%S")?;
            
            let now = Local::now().naive_local();
            if start_time > now {
                let wait_duration = start_time.signed_duration_since(now).to_std()?;
                println!("⏳ Waiting until {} before starting the polling...", start_time);
                sleep(wait_duration).await;
            }

            let config = LambdaConfig::from_env();

            println!("🚀 Starting Captcha-GPU randomized polling (Max Hourly Budget: {})...", max_hourly_budget);
            let mut rng = rand::thread_rng();

            for block in &selected_strategy.blocks {
                let block_start = NaiveDateTime::parse_from_str(&block.block_start, "%Y-%m-%d %H:%M:%S")?;
                let block_end = NaiveDateTime::parse_from_str(&block.block_end, "%Y-%m-%d %H:%M:%S")?;
                
                let now = Local::now().naive_local();
                if block_end < now {
                    continue;
                }
                
                if block_start > now {
                    let wait = block_start.signed_duration_since(now).to_std()?;
                    println!("⏳ Waiting {}s for block {} to start...", wait.as_secs(), block_start);
                    sleep(wait).await;
                }

                // Dynamic Budgeting logic
                let elapsed_minutes = (Local::now().naive_local() - start_time).num_minutes();
                let hourly_budget = if elapsed_minutes < 45 {
                    (max_hourly_budget as f64 * 0.8) as u64
                } else {
                    max_hourly_budget
                };

                // CAP the block budget to respect 1 req/sec limit (900 reqs per 15 min block)
                let block_duration_secs = 15.0 * 60.0;
                let max_safe_block_budget = (block_duration_secs * 0.95) as u64; // 855 requests max per 15m
                
                let block_budget = std::cmp::min(
                    max_safe_block_budget,
                    ((hourly_budget as f64) * (block.budget_percent / 100.0)) as u64
                );

                if block_budget == 0 {
                    continue;
                }

                let target_interval = block_duration_secs / (block_budget as f64);
                let jitter_range = target_interval * 0.3;
                
                println!("📦 Block {} to {}: Budget {} requests (Avg interval: {:.2}s)", 
                    block_start, block_end, block_budget, target_interval);

                for i in 0..block_budget {
                    if Local::now().naive_local() > block_end {
                        println!("⏰ Block time ended.");
                        break;
                    }

                    println!("[{}] 🔍 (Req {}/{}) Polling API for {}...", Local::now().format("%H:%M:%S"), i+1, block_budget, gpu);
                    
                    match client.get_instance_types().await {
                        Ok(response) => {
                            let mut found = false;
                            for (name, availability) in response.data {
                                let name_lower = name.to_lowercase();
                                let target_lower = gpu.to_lowercase();
                                let matches = if target_lower == "a10" {
                                    name_lower.contains("a10") && !name_lower.contains("a100")
                                } else {
                                    name_lower.contains(&target_lower)
                                };

                                if matches && !availability.regions_with_capacity_available.is_empty() {
                                    found = true;
                                    let region = availability.regions_with_capacity_available[0].name.clone();
                                    println!("🚨 BINGO! {} IS AVAILABLE in {}! 🚨", name, region);
                                    
                                    if dry_run {
                                        println!("[DRY RUN] Would launch {} in {}", name, region);
                                        return Ok(());
                                    } else {
                                        match launch_and_setup(&client, &config, &name, &gpu, &region).await {
                                            Ok(_) => return Ok(()),
                                            Err(e) => eprintln!("❌ Launch failed: {}", e),
                                        }
                                    }
                                }
                            }
                            if !found {
                                println!("   ❌ Not available yet.");
                            }
                        }
                        Err(e) => {
                            let err_msg = e.to_string();
                            eprintln!("❌ API Error: {}", err_msg);
                            if err_msg.contains("429") {
                                println!("⚠️  Rate limit hit! Cooling down for 5 seconds...");
                                sleep(Duration::from_secs(5)).await;
                            }
                        }
                    }

                    let sleep_time = target_interval + rng.gen_range(-jitter_range..jitter_range);
                    if sleep_time > 0.0 {
                        sleep(Duration::from_secs_f64(sleep_time)).await;
                    }
                }
            }
            println!("🏁 Timeslot polling finished.");
        }
        Commands::Stats { types, min_interval, max_interval, output } => {
            let target_types = types.unwrap_or_else(|| vec!["a10".to_string(), "a100".to_string(), "gh200".to_string()]);
            println!("📊 Collecting statistics for: {:?}", target_types);
            println!("📁 Saving to: {}", output);
            println!("⏱️  Sampling interval: {}-{} seconds (randomized)", min_interval, max_interval);
            println!("Press Ctrl+C to stop.\n");

            let mut file_exists = Path::new(&output).exists();
            let mut rng = rand::thread_rng();

            loop {
                let now = Local::now();
                let timestamp = now.format("%Y-%m-%d %H:%M:%S").to_string();

                match client.get_instance_types().await {
                    Ok(response) => {
                        let is_file_empty = file_exists && Path::new(&output).metadata()?.len() == 0;
                        let file = OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&output)?;
                        
                        let mut wtr = WriterBuilder::new()
                            .has_headers(!file_exists || is_file_empty)
                            .from_writer(file);

                        if !file_exists || is_file_empty {
                            wtr.write_record(&["timestamp", "type", "available", "region_count", "regions"])?;
                            file_exists = true; // Mark as existing now that we wrote headers
                        }

                        println!("[{}] Sampling...", timestamp);
                        let mut recorded_names = std::collections::HashSet::new();

                        for target in &target_types {
                            let target_lower = target.to_lowercase();
                            for (name, availability) in &response.data {
                                let name_lower = name.to_lowercase();
                                
                                // Precise match logic: a10 should not match a100
                                let matches = if target_lower == "a10" {
                                    name_lower.contains("a10") && !name_lower.contains("a100")
                                } else {
                                    name_lower.contains(&target_lower)
                                };

                                if matches && !recorded_names.contains(name) {
                                    recorded_names.insert(name.clone());
                                    let region_count = availability.regions_with_capacity_available.len();
                                    let is_available = region_count > 0;
                                    let regions = availability.regions_with_capacity_available
                                        .iter()
                                        .map(|r| r.name.clone())
                                        .collect::<Vec<_>>()
                                        .join("|");

                                    wtr.write_record(&[
                                        &timestamp,
                                        name,
                                        &is_available.to_string(),
                                        &region_count.to_string(),
                                        &regions,
                                    ])?;
                                    
                                    let status_char = if is_available { "✅" } else { "❌" };
                                    println!("  {} {:<10} | Regions: {}", status_char, name, region_count);
                                }
                            }
                        }
                        wtr.flush()?;
                    }
                    Err(e) => eprintln!("[{}] ❌ Error sampling API: {}", timestamp, e),
                }

                let wait_secs = rng.gen_range(min_interval..=max_interval);
                println!("... next sample in {}s\n", wait_secs);
                sleep(Duration::from_secs(wait_secs)).await;
            }
        }
        Commands::Monitor { types, interval } => {
            let target_types = types.unwrap_or_else(|| vec!["a10".to_string(), "a100".to_string(), "gh200".to_string()]);
            println!("👀 Monitoring availability for: {:?}", target_types);
            println!("⏱️  Interval: {} seconds", interval);
            println!("Press Ctrl+C to stop.\n");

            let mut sample_count = 0;
            loop {
                sample_count += 1;
                if sample_count >= 500 {
                    println!("🔄 Collecting 500 samples. Triggering retraining...");
                    let status = std::process::Command::new("python3")
                        .arg("retrain_models.py")
                        .status();
                    
                    match status {
                        Ok(s) if s.success() => println!("✅ Retraining complete."),
                        Ok(s) => eprintln!("❌ Retraining failed with exit code: {}", s),
                        Err(e) => eprintln!("❌ Failed to start retraining process: {}", e),
                    }
                    sample_count = 0;
                }

                match client.get_instance_types().await {
                    Ok(response) => {
                        for (name, availability) in response.data {
                            let name_lower = name.to_lowercase();
                            let is_match = target_types.iter().any(|t| {
                                let t_lower = t.to_lowercase();
                                if t_lower == "a10" && name_lower.contains("a100") {
                                    return false;
                                }
                                name_lower.contains(&t_lower)
                            });

                            if is_match {
                                let available_regions = &availability.regions_with_capacity_available;
                                if !available_regions.is_empty() {
                                    let price = availability.instance_type.price_cents_per_hour.unwrap_or(0);
                                    println!("\n🔔 ALERT: {} is AVAILABLE in:", name);
                                    println!("   💰 Price: ${:.2}/hour", price as f64 / 100.0);
                                    for region in available_regions {
                                        println!("   - {} ({})", region.name, region.description);
                                    }

                                    if Confirm::new()
                                        .with_prompt(format!("Do you want to launch a {} instance?", name))
                                        .default(false)
                                        .interact()?
                                    {
                                        // 1. Select Region
                                        let region_options: Vec<String> = available_regions.iter().map(|r| r.name.clone()).collect();
                                        let region_idx = Select::new()
                                            .with_prompt("Select a region")
                                            .items(&region_options)
                                            .default(0)
                                            .interact()?;
                                        let selected_region = region_options[region_idx].clone();

                                        let config = LambdaConfig::from_env();
                                        let gpu_type = target_types.iter()
                                            .find(|t| name.to_lowercase().contains(&t.to_lowercase()))
                                            .cloned()
                                            .unwrap_or_else(|| name.clone());
                                            
                                        match launch_and_setup(&client, &config, &name, &gpu_type, &selected_region).await {
                                            Ok(_) => return Ok(()),
                                            Err(e) => eprintln!("❌ Launch failed: {}", e),
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("❌ Error checking availability: {}", e),
                }
                sleep(Duration::from_secs(interval)).await;
            }
        }
        Commands::Check { types } => {
            let target_types = types.unwrap_or_else(|| vec!["a10".to_string(), "a100".to_string(), "gh200".to_string()]);
            println!("🔍 Checking availability for: {:?}\n", target_types);

            let response = client.get_instance_types().await?;
            let mut found = false;

            for (name, availability) in response.data {
                let name_lower = name.to_lowercase();
                let is_match = target_types.iter().any(|t| {
                    let t_lower = t.to_lowercase();
                    if t_lower == "a10" && name_lower.contains("a100") {
                        return false;
                    }
                    name_lower.contains(&t_lower)
                });

                if is_match {
                    let available_regions = &availability.regions_with_capacity_available;
                    if !available_regions.is_empty() {
                        found = true;
                        let price = availability.instance_type.price_cents_per_hour.unwrap_or(0);
                        println!("✅ {} is AVAILABLE in:", name);
                        println!("   💰 Price: ${:.2}/hour", price as f64 / 100.0);
                        for region in available_regions {
                            println!("   - {} ({})", region.name, region.description);
                        }
                    }
                }
            }

            if !found {
                println!("No target instances are currently available.");
            }
        }
        Commands::Instances => {
            println!("🔍 Checking your Lambda Cloud instances...\n");
            let instances_response = client.get_instances().await?;
            let instances = instances_response.data;
            if instances.is_empty() {
                println!("✅ No instances found. You have 0 launched instances.");
            } else {
                println!("📊 Found {} launched instance(s):\n", instances.len());
                for (i, instance) in instances.iter().enumerate() {
                    println!("Instance {}:", i + 1);
                    println!("  ID: {}", instance.id);
                    println!("  Name: {}", instance.name.as_deref().unwrap_or("Unnamed"));
                    println!("  Status: {}", instance.status);
                    if let Some(ip) = &instance.ip {
                        println!("  Public IP: {}", ip);
                    }
                    if let Some(pip) = &instance.private_ip {
                        println!("  Private IP: {}", pip);
                    }
                    if let Some(instance_type) = &instance.instance_type {
                        println!("  Type: {}", instance_type.name);
                        if let Some(price) = instance_type.price_cents_per_hour {
                            println!("  Price: ${:.2}/hour", price as f64 / 100.0);
                        }
                    }
                    if let Some(region) = &instance.region {
                        println!("  Region: {}", region);
                    }
                    if let Some(created) = &instance.created {
                        println!("  Created: {}", created);
                    }
                    println!();
                }
            }
        }
        Commands::Types => {
            println!("📋 Listing all instance types and their status:\n");
            let response = client.get_instance_types().await?;
            let mut types: Vec<_> = response.data.into_iter().collect();
            types.sort_by(|a, b| a.0.cmp(&b.0));

            for (name, availability) in types {
                let status = if availability.regions_with_capacity_available.is_empty() {
                    "❌ UNAVAILABLE"
                } else {
                    "✅ AVAILABLE  "
                };
                println!("{} | {}", status, name);
            }
        }
    }

    Ok(())
}
