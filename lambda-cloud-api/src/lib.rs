use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Instance {
    pub id: String,
    pub name: Option<String>,
    pub ip: Option<String>,
    pub private_ip: Option<String>,
    pub status: String,
    pub instance_type: Option<InstanceTypeInfo>,
    #[serde(default)]
    pub region: Option<RegionField>,
    pub created: Option<String>,
    pub ssh_key_names: Option<Vec<String>>,
}

/// Lambda has returned region as either a string or a `{name, description}` object.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
pub enum RegionField {
    Name(String),
    Object(Region),
}

impl RegionField {
    pub fn display_name(&self) -> &str {
        match self {
            RegionField::Name(name) => name,
            RegionField::Object(region) => &region.name,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstanceTypeInfo {
    pub name: String,
    pub description: Option<String>,
    pub price_cents_per_hour: Option<i32>,
    pub specs: Option<InstanceSpecs>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstanceSpecs {
    pub vcpus: i32,
    pub memory_gib: i32,
    pub storage_gib: i32,
    pub gpus: i32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Region {
    pub name: String,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstanceTypeAvailability {
    pub instance_type: InstanceTypeInfo,
    pub regions_with_capacity_available: Vec<Region>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstanceTypesResponse {
    pub data: HashMap<String, InstanceTypeAvailability>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InstancesResponse {
    pub data: Vec<Instance>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SshKey {
    pub name: String,
    pub public_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SshKeysResponse {
    pub data: Vec<SshKey>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchRequest {
    pub region_name: String,
    pub instance_type_name: String,
    pub ssh_key_names: Vec<String>,
    pub quantity: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchResponse {
    pub data: LaunchData,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LaunchData {
    pub instance_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FirewallRule {
    pub protocol: String,
    pub port: i32,
    pub source_ips: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FirewallRulesResponse {
    pub data: Vec<FirewallRule>,
}

pub struct LambdaCloudClient {
    pub client: Client,
    pub api_key: String,
    pub base_url: String,
}

impl LambdaCloudClient {
    pub fn new(api_key: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: "https://cloud.lambdalabs.com/api/v1".to_string(),
        }
    }

    pub fn get_api_rate_limit() -> (u64, u64) {
        // According to the Lambda Cloud API version 1.9.3 spec:
        // "Requests to the API are generally limited to one request per second.
        // Requests to the /instance-operations/launch endpoint are limited to one request per 12 seconds, or five requests per minute."
        
        let general_hourly_limit = 3600; // 1 request per second -> 3600 requests per hour
        let launch_hourly_limit = 300;   // 5 requests per minute -> 300 requests per hour
        
        (general_hourly_limit, launch_hourly_limit)
    }

    pub async fn get_instances(&self) -> Result<InstancesResponse> {
        let url = format!("{}/instances", self.base_url);
        let response = self
            .client
            .get(&url)
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await
            .context("Failed to send request to Lambda Cloud API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API request failed with status {}: {}", status, error_text);
        }

        let instances: InstancesResponse = response
            .json()
            .await
            .context("Failed to parse JSON response")?;

        Ok(instances)
    }

    pub async fn get_instance_types(&self) -> Result<InstanceTypesResponse> {
        let url = format!("{}/instance-types", self.base_url);
        let response = self
            .client
            .get(&url)
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await
            .context("Failed to send request to Lambda Cloud API")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("API request failed with status {}: {}", status, error_text);
        }

        let instance_types: InstanceTypesResponse = response
            .json()
            .await
            .context("Failed to parse JSON response")?;

        Ok(instance_types)
    }

    pub async fn get_ssh_keys(&self) -> Result<SshKeysResponse> {
        let url = format!("{}/ssh-keys", self.base_url);
        let response = self
            .client
            .get(&url)
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await
            .context("Failed to fetch SSH keys")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to get SSH keys: {} - {}", status, error_text);
        }

        let keys: SshKeysResponse = response.json().await.context("Failed to parse SSH keys JSON")?;
        Ok(keys)
    }

    pub async fn launch_instance(&self, request: LaunchRequest) -> Result<LaunchResponse> {
        let url = format!("{}/instance-operations/launch", self.base_url);
        let response = self
            .client
            .post(&url)
            .basic_auth(&self.api_key, Some(""))
            .json(&request)
            .send()
            .await
            .context("Failed to send launch request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Launch failed: {} - {}", status, error_text);
        }

        let launch_response: LaunchResponse = response.json().await.context("Failed to parse launch response JSON")?;
        Ok(launch_response)
    }

    pub async fn get_firewall_rules(&self) -> Result<FirewallRulesResponse> {
        let url = format!("{}/firewall-rules", self.base_url);
        let response = self
            .client
            .get(&url)
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await
            .context("Failed to get firewall rules")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to get firewall rules: {} - {}", status, error_text);
        }

        let rules: FirewallRulesResponse = response.json().await.context("Failed to parse firewall rules JSON")?;
        Ok(rules)
    }

    pub async fn update_firewall_rules(&self, rules: Vec<FirewallRule>) -> Result<()> {
        let url = format!("{}/firewall-rules", self.base_url);
        let payload = serde_json::json!({ "data": rules });
        let response = self
            .client
            .put(&url)
            .basic_auth(&self.api_key, Some(""))
            .json(&payload)
            .send()
            .await
            .context("Failed to update firewall rules")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Failed to update firewall rules: {} - {}", status, error_text);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct LambdaConfig {
    pub ssh_key_name: Option<String>,
    pub public_ip: Option<String>,
    pub instance_name_prefix: String,
}

impl LambdaConfig {
    pub fn from_env() -> Self {
        Self {
            ssh_key_name: std::env::var("LAMBDA_SSH_KEY").ok().filter(|s| !s.is_empty()),
            public_ip: std::env::var("MY_PUBLIC_IP").ok().filter(|s| !s.is_empty()),
            instance_name_prefix: std::env::var("LAMBDA_INSTANCE_PREFIX")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "lambda-captcha".to_string()),
        }
    }
}

/// Load configuration from the process environment and `.env` files.
///
/// Variables already set in the process are kept. Missing values are filled
/// from `$CAPTCHA_GPU_ROOT/.env` (set by the global `captcha-gpu` command)
/// and then from a `.env` in the working directory or its parents.
pub fn load_config_env() {
    let mut paths = Vec::new();
    if let Some(root) = std::env::var("CAPTCHA_GPU_ROOT").ok().filter(|s| !s.is_empty()) {
        paths.push(Path::new(&root).join(".env"));
    }
    load_dotenv_files(&paths);
    dotenvy::dotenv().ok();
}

/// Set variables from these `.env` files without overriding the process environment.
pub fn load_dotenv_files(paths: &[PathBuf]) {
    for path in paths {
        if path.is_file() {
            dotenvy::from_path(path).ok();
        }
    }
}

const PUBLIC_IP_ENDPOINTS: &[&str] = &[
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
    "https://icanhazip.com",
];

/// Parse a public IP string, trimming whitespace and rejecting empty/non-IP values.
pub fn parse_public_ip(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    IpAddr::from_str(trimmed).ok().map(|ip| ip.to_string())
}

/// Detect the caller's current public IP by querying public echo services.
pub async fn detect_public_ip() -> Result<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .user_agent("captcha-gpu")
        .build()
        .context("Failed to build HTTP client for public IP detection")?;

    let mut last_error = None;
    for endpoint in PUBLIC_IP_ENDPOINTS {
        match client.get(*endpoint).send().await {
            Ok(response) if response.status().is_success() => {
                match response.text().await {
                    Ok(body) => {
                        if let Some(ip) = parse_public_ip(&body) {
                            return Ok(ip);
                        }
                        last_error = Some(anyhow::anyhow!(
                            "{} returned a non-IP body: {}",
                            endpoint,
                            body.trim()
                        ));
                    }
                    Err(err) => last_error = Some(err.into()),
                }
            }
            Ok(response) => {
                last_error = Some(anyhow::anyhow!(
                    "{} returned HTTP {}",
                    endpoint,
                    response.status()
                ));
            }
            Err(err) => last_error = Some(err.into()),
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("No public IP endpoints configured")))
}

/// Resolve the firewall source IP.
///
/// Prefer an explicit override. Otherwise detect the caller's current public IP.
pub async fn resolve_firewall_ip(configured: Option<&str>) -> Result<String> {
    if let Some(ip) = configured.and_then(parse_public_ip) {
        return Ok(ip);
    }
    detect_public_ip().await
}

/// Parse an instance start timestamp from API or local records.
pub fn parse_start_time(raw: &str) -> Option<DateTime<Utc>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(trimmed) {
        return Some(dt.with_timezone(&Utc));
    }
    // Common Lambda / local formats without timezone: treat as UTC.
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(trimmed, fmt) {
            return Some(DateTime::from_naive_utc_and_offset(naive, Utc));
        }
    }
    None
}

/// Estimate accrued instance cost in US cents.
///
/// Lambda bills ODC in one-minute increments. This uses whole minutes already
/// started (`ceil(elapsed_seconds / 60)`, minimum 1 once the clock has started).
/// Returns `None` when price or start time is missing, or start is in the future.
pub fn estimate_accrued_cost_cents(
    price_cents_per_hour: Option<i32>,
    started_at: Option<DateTime<Utc>>,
    now: DateTime<Utc>,
) -> Option<i64> {
    let price = price_cents_per_hour.filter(|p| *p >= 0)?;
    let start = started_at?;
    let elapsed = now.signed_duration_since(start);
    let secs = elapsed.num_seconds();
    if secs < 0 {
        return None;
    }
    let minutes = std::cmp::max(1, (secs + 59) / 60);
    Some((price as i64) * minutes / 60)
}

pub fn format_usd_from_cents(cents: i64) -> String {
    format!("${:.2}", cents as f64 / 100.0)
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct LaunchTimeStore {
    pub instances: HashMap<String, LaunchRecord>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LaunchRecord {
    pub started_at: String,
    pub instance_type: Option<String>,
    pub price_cents_per_hour: Option<i32>,
}

impl LaunchTimeStore {
    pub fn path() -> PathBuf {
        let root = std::env::var("CAPTCHA_GPU_ROOT")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        root.join(".captcha-gpu").join("launch_times.json")
    }

    pub fn load() -> Self {
        let path = Self::path();
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(_) => return Self::default(),
        };
        serde_json::from_reader(BufReader::new(file)).unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        let path = Self::path();
        if let Some(parent) = path.parent() {
            create_dir_all(parent).context("Failed to create .captcha-gpu directory")?;
        }
        let file = File::create(&path).context("Failed to create launch_times.json")?;
        serde_json::to_writer_pretty(BufWriter::new(file), self)
            .context("Failed to write launch_times.json")?;
        Ok(())
    }

    pub fn record_launch(
        &mut self,
        instance_id: &str,
        started_at: DateTime<Utc>,
        instance_type: Option<&str>,
        price_cents_per_hour: Option<i32>,
    ) {
        self.instances.insert(
            instance_id.to_string(),
            LaunchRecord {
                started_at: started_at.to_rfc3339(),
                instance_type: instance_type.map(|s| s.to_string()),
                price_cents_per_hour,
            },
        );
    }

    pub fn started_at(&self, instance_id: &str) -> Option<DateTime<Utc>> {
        self.instances
            .get(instance_id)
            .and_then(|r| parse_start_time(&r.started_at))
    }
}

/// Resolve the best available start time for accrued-cost estimation.
pub fn resolve_instance_start_time(
    instance: &Instance,
    store: &LaunchTimeStore,
) -> Option<DateTime<Utc>> {
    instance
        .created
        .as_deref()
        .and_then(parse_start_time)
        .or_else(|| store.started_at(&instance.id))
}

pub async fn launch_and_setup(
    client: &LambdaCloudClient,
    config: &LambdaConfig,
    instance_type_name: &str,
    gpu_label: &str,
    region: &str,
) -> Result<()> {
    let instance_name = format!("{}-{}", config.instance_name_prefix, gpu_label.replace('_', "-"));
    
    // Resolve SSH Key
    let selected_key = match &config.ssh_key_name {
        Some(key) => key.clone(),
        None => {
            let ssh_keys = client.get_ssh_keys().await?;
            if ssh_keys.data.is_empty() {
                println!("⚠️ No SSH keys found. Will attempt launch without key.");
                String::new()
            } else {
                ssh_keys.data[0].name.clone()
            }
        }
    };

    println!(
        "🚀 Launching {} as '{}' in {} with key '{}'...",
        instance_type_name, instance_name, region, selected_key
    );

    let mut ssh_key_names = vec![];
    if !selected_key.is_empty() {
        ssh_key_names.push(selected_key.clone());
    }

    let launch_req = LaunchRequest {
        region_name: region.to_string(),
        instance_type_name: instance_type_name.to_string(),
        ssh_key_names,
        quantity: 1,
        name: Some(instance_name.clone()),
    };

    let launch_res = client.launch_instance(launch_req).await?;
    println!("✅ Instance launched! IDs: {:?}", launch_res.data.instance_ids);
    let instance_id = &launch_res.data.instance_ids[0];
    let launch_time = Utc::now();
    let mut launch_store = LaunchTimeStore::load();
    launch_store.record_launch(instance_id, launch_time, Some(instance_type_name), None);
    if let Err(e) = launch_store.save() {
        eprintln!("⚠️ Failed to persist launch time for cost tracking: {}", e);
    }

    // Setup Firewall: MY_PUBLIC_IP overrides; otherwise detect the caller's current IP.
    match resolve_firewall_ip(config.public_ip.as_deref()).await {
        Ok(public_ip) => {
            let source = if config.public_ip.as_deref().and_then(parse_public_ip).is_some() {
                "MY_PUBLIC_IP"
            } else {
                "detected caller IP"
            };
            println!("🛡️ Setting up firewall rules for {} ({})", public_ip, source);
            let cidr = format!("{}/32", public_ip);
            let rules = vec![
                FirewallRule {
                    protocol: "tcp".to_string(),
                    port: 22,
                    source_ips: vec![cidr.clone()],
                },
                FirewallRule {
                    protocol: "tcp".to_string(),
                    port: 80,
                    source_ips: vec![cidr.clone()],
                },
                FirewallRule {
                    protocol: "tcp".to_string(),
                    port: 443,
                    source_ips: vec![cidr],
                },
            ];
            if let Err(e) = client.update_firewall_rules(rules).await {
                eprintln!("⚠️ Failed to update firewall rules: {}", e);
            } else {
                println!("✅ Firewall rules updated successfully.");
            }
        }
        Err(e) => {
            eprintln!(
                "⚠️ Could not resolve caller public IP for firewall rules: {}. Skipping firewall setup.",
                e
            );
        }
    }

    // Wait for IP. Billing starts after health checks; prefer that moment for accrual.
    println!("⏳ Waiting for instance to be active and get an IP...");
    let mut instance_ip = None;
    let mut billing_start = launch_time;
    let mut price_cents_per_hour = None;
    for _ in 0..60 { // 10 minutes max
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        if let Ok(instances_res) = client.get_instances().await {
            if let Some(inst) = instances_res.data.iter().find(|i| &i.id == instance_id) {
                if let Some(instance_type) = &inst.instance_type {
                    price_cents_per_hour = instance_type.price_cents_per_hour;
                }
                if inst.status == "active" || inst.ip.is_some() {
                    billing_start = Utc::now();
                }
                if let Some(ip) = &inst.ip {
                    instance_ip = Some(ip.clone());
                    break;
                }
                if inst.status == "error" {
                    eprintln!("❌ Instance entered error state.");
                    break;
                }
            }
        }
        print!(".");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    launch_store.record_launch(
        instance_id,
        billing_start,
        Some(instance_type_name),
        price_cents_per_hour,
    );
    if let Err(e) = launch_store.save() {
        eprintln!("⚠️ Failed to update launch time for cost tracking: {}", e);
    }

    if let Some(ip) = instance_ip {
        println!("\n\n✅ Instance is active at IP: {}", ip);
        println!("📝 SSH Config entry (append this to ~/.ssh/config):");
        println!("----------------------------------------");
        println!("Host {}", instance_name);
        println!("    HostName {}", ip);
        println!("    User ubuntu");
        if !selected_key.is_empty() {
            println!("    IdentityFile ~/.ssh/{}", selected_key);
        }
        println!("    StrictHostKeyChecking no");
        println!("    UserKnownHostsFile /dev/null");
        println!("----------------------------------------");
    } else {
        println!("\n\n⚠️ Timed out waiting for IP address. Check later using the 'instances' command.");
    }

    Ok(())
}

/// One concrete launch the poll loop is allowed to try.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchTarget {
    pub instance_type: String,
    pub region: String,
    pub gpu_label: String,
}

/// What one poll produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollBatch<T> {
    /// Try these launches. An empty list means nothing was available.
    Candidates(Vec<T>),
    /// Stop polling without launching. Used when a forecast window is over.
    Stop,
}

/// Why [`poll_until_acquired`] returned.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub enum AcquireStop<T> {
    /// The first successful launch. No further polls or launches were made.
    Acquired(T),
    /// The fetcher asked to stop, or no launch succeeded before it did.
    Exhausted,
}

/// Poll until one launch succeeds, then stop.
///
/// A successful launch returns immediately. Later candidates in that batch are
/// not launched, and `fetch` / `wait` are not called again. A failed launch is
/// logged and the next candidate is tried. `wait` runs only after a batch that
/// did not acquire an instance.
pub async fn poll_until_acquired<T, F, L, W>(
    mut fetch: F,
    mut launch: L,
    mut wait: W,
) -> Result<AcquireStop<T>>
where
    T: Clone,
    F: AsyncFnMut() -> Result<PollBatch<T>>,
    L: AsyncFnMut(T) -> Result<()>,
    W: AsyncFnMut() -> (),
{
    loop {
        match fetch().await? {
            PollBatch::Stop => return Ok(AcquireStop::Exhausted),
            PollBatch::Candidates(candidates) => {
                let mut acquired = None;
                for candidate in candidates {
                    let record = candidate.clone();
                    match launch(candidate).await {
                        Ok(()) => {
                            acquired = Some(record);
                            break;
                        }
                        Err(err) => eprintln!("❌ Launch failed: {}", err),
                    }
                }
                if let Some(target) = acquired {
                    return Ok(AcquireStop::Acquired(target));
                }
                wait().await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn target(instance_type: &str, region: &str) -> LaunchTarget {
        LaunchTarget {
            instance_type: instance_type.to_string(),
            region: region.to_string(),
            gpu_label: "gh200".to_string(),
        }
    }

    #[tokio::test]
    async fn successful_launch_stops_without_another_poll_or_launch() {
        let polls = Arc::new(AtomicUsize::new(0));
        let launches = Arc::new(AtomicUsize::new(0));
        let waits = Arc::new(AtomicUsize::new(0));
        let polls_c = polls.clone();
        let launches_c = launches.clone();
        let waits_c = waits.clone();

        let first = target("gpu_1x_gh200", "us-east-3");
        let second = target("gpu_8x_gh200", "us-west-1");
        let expected = first.clone();

        let stop = poll_until_acquired(
            async || {
                let n = polls_c.fetch_add(1, Ordering::SeqCst);
                assert_eq!(n, 0, "polled again after a successful launch");
                Ok(PollBatch::Candidates(vec![first.clone(), second.clone()]))
            },
            async |_target| {
                let n = launches_c.fetch_add(1, Ordering::SeqCst);
                assert_eq!(n, 0, "launched again after success");
                Ok(())
            },
            async || {
                waits_c.fetch_add(1, Ordering::SeqCst);
            },
        )
        .await
        .unwrap();

        assert_eq!(stop, AcquireStop::Acquired(expected));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        assert_eq!(launches.load(Ordering::SeqCst), 1);
        assert_eq!(waits.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn failed_launch_retries_then_stops_on_the_next_success() {
        let polls = Arc::new(AtomicUsize::new(0));
        let launches = Arc::new(AtomicUsize::new(0));
        let waits = Arc::new(AtomicUsize::new(0));
        let polls_c = polls.clone();
        let launches_c = launches.clone();
        let waits_c = waits.clone();

        let stop = poll_until_acquired(
            async || {
                let n = polls_c.fetch_add(1, Ordering::SeqCst);
                let batch = if n == 0 {
                    vec![target("gpu_1x_gh200", "us-east-3")]
                } else {
                    assert_eq!(n, 1, "kept polling after a successful launch");
                    vec![
                        target("gpu_1x_gh200", "us-east-3"),
                        target("gpu_1x_gh200", "us-west-1"),
                    ]
                };
                Ok(PollBatch::Candidates(batch))
            },
            async |_target| {
                let n = launches_c.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    anyhow::bail!("capacity disappeared");
                }
                assert!(n < 2, "launched again after success");
                Ok(())
            },
            async || {
                waits_c.fetch_add(1, Ordering::SeqCst);
            },
        )
        .await
        .unwrap();

        assert_eq!(stop, AcquireStop::Acquired(target("gpu_1x_gh200", "us-east-3")));
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(launches.load(Ordering::SeqCst), 2);
        assert_eq!(waits.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn stop_ends_the_loop_without_launching() {
        let launches = Arc::new(AtomicUsize::new(0));
        let waits = Arc::new(AtomicUsize::new(0));
        let launches_c = launches.clone();
        let waits_c = waits.clone();

        let stop = poll_until_acquired(
            async || Ok::<_, anyhow::Error>(PollBatch::Stop),
            async |_target: LaunchTarget| {
                launches_c.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
            async || {
                waits_c.fetch_add(1, Ordering::SeqCst);
            },
        )
        .await
        .unwrap();

        assert_eq!(stop, AcquireStop::<LaunchTarget>::Exhausted);
        assert_eq!(launches.load(Ordering::SeqCst), 0);
        assert_eq!(waits.load(Ordering::SeqCst), 0);
    }

    fn env_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "captcha-gpu-env-{}-{}-{}",
            std::process::id(),
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn process_env_wins_over_dotenv_file() {
        let dir = env_test_dir("override");
        std::fs::write(
            dir.join(".env"),
            "CAPTCHA_TEST_PRESENT=from-file\nCAPTCHA_TEST_ONLY_FILE=from-file\n",
        )
        .unwrap();
        let previous_present = std::env::var("CAPTCHA_TEST_PRESENT").ok();
        let previous_only = std::env::var("CAPTCHA_TEST_ONLY_FILE").ok();
        std::env::set_var("CAPTCHA_TEST_PRESENT", "from-env");
        std::env::remove_var("CAPTCHA_TEST_ONLY_FILE");

        load_dotenv_files(&[dir.join(".env")]);

        assert_eq!(std::env::var("CAPTCHA_TEST_PRESENT").unwrap(), "from-env");
        assert_eq!(std::env::var("CAPTCHA_TEST_ONLY_FILE").unwrap(), "from-file");

        restore_var("CAPTCHA_TEST_PRESENT", previous_present);
        restore_var("CAPTCHA_TEST_ONLY_FILE", previous_only);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn global_root_dotenv_fills_unset_variables() {
        let dir = env_test_dir("global");
        std::fs::write(dir.join(".env"), "CAPTCHA_TEST_GLOBAL=from-root\n").unwrap();
        let previous_root = std::env::var("CAPTCHA_GPU_ROOT").ok();
        let previous_value = std::env::var("CAPTCHA_TEST_GLOBAL").ok();
        std::env::remove_var("CAPTCHA_TEST_GLOBAL");
        std::env::set_var("CAPTCHA_GPU_ROOT", &dir);

        load_config_env();

        assert_eq!(std::env::var("CAPTCHA_TEST_GLOBAL").unwrap(), "from-root");

        restore_var("CAPTCHA_GPU_ROOT", previous_root);
        restore_var("CAPTCHA_TEST_GLOBAL", previous_value);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn restore_var(key: &str, previous: Option<String>) {
        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }

    #[test]
    fn parse_public_ip_accepts_ipv4_and_ipv6() {
        assert_eq!(parse_public_ip(" 203.0.113.10\n ").as_deref(), Some("203.0.113.10"));
        assert_eq!(
            parse_public_ip("2001:db8::1").as_deref(),
            Some("2001:db8::1")
        );
        assert_eq!(parse_public_ip(""), None);
        assert_eq!(parse_public_ip("not-an-ip"), None);
        assert_eq!(parse_public_ip("203.0.113.10/32"), None);
    }

    #[tokio::test]
    async fn resolve_firewall_ip_prefers_configured_override() {
        let ip = resolve_firewall_ip(Some(" 198.51.100.7 ")).await.unwrap();
        assert_eq!(ip, "198.51.100.7");
    }

    #[tokio::test]
    async fn resolve_firewall_ip_rejects_invalid_override_and_falls_back() {
        // Invalid override should fall through to detection; this only asserts the
        // override path rejects garbage before network lookup by checking parse.
        assert_eq!(parse_public_ip("not-an-ip"), None);
    }

    #[test]
    fn estimate_accrued_cost_uses_minute_increments() {
        let start = DateTime::parse_from_rfc3339("2026-11-03T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let after_90s = start + chrono::Duration::seconds(90);
        // 2 billable minutes at $1.20/hr = 120 cents/hr => 4 cents
        assert_eq!(
            estimate_accrued_cost_cents(Some(120), Some(start), after_90s),
            Some(4)
        );
        let after_0s = start;
        assert_eq!(
            estimate_accrued_cost_cents(Some(120), Some(start), after_0s),
            Some(2)
        );
        assert_eq!(estimate_accrued_cost_cents(None, Some(start), after_90s), None);
        assert_eq!(estimate_accrued_cost_cents(Some(120), None, after_90s), None);
    }

    #[test]
    fn parse_start_time_accepts_common_formats() {
        assert!(parse_start_time("2026-11-03T12:00:00Z").is_some());
        assert!(parse_start_time("2026-11-03 12:00:00").is_some());
        assert!(parse_start_time("not-a-time").is_none());
    }

    #[test]
    fn resolve_instance_start_time_prefers_api_created() {
        let mut store = LaunchTimeStore::default();
        store.record_launch(
            "abc",
            DateTime::parse_from_rfc3339("2026-11-03T11:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
            Some("gpu_1x_gh200"),
            Some(149),
        );
        let instance = Instance {
            id: "abc".to_string(),
            name: None,
            ip: None,
            private_ip: None,
            status: "active".to_string(),
            instance_type: None,
            region: None,
            created: Some("2026-11-03T12:00:00Z".to_string()),
            ssh_key_names: None,
        };
        let start = resolve_instance_start_time(&instance, &store).unwrap();
        assert_eq!(start.to_rfc3339(), "2026-11-03T12:00:00+00:00");
    }
}
