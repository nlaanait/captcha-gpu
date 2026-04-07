use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Instance {
    pub id: String,
    pub name: Option<String>,
    pub ip: Option<String>,
    pub private_ip: Option<String>,
    pub status: String,
    pub instance_type: Option<InstanceTypeInfo>,
    pub region: Option<String>,
    pub created: Option<String>,
    pub ssh_key_names: Option<Vec<String>>,
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
                .unwrap_or_else(|_| "lambda-captcha".to_string()),
        }
    }
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

    // Setup Firewall
    if let Some(public_ip) = &config.public_ip {
        println!("🛡️ Setting up firewall rules for IP: {}", public_ip);
        let rules = vec![
            FirewallRule {
                protocol: "tcp".to_string(),
                port: 22,
                source_ips: vec![format!("{}/32", public_ip)],
            },
            FirewallRule {
                protocol: "tcp".to_string(),
                port: 80,
                source_ips: vec![format!("{}/32", public_ip)],
            },
            FirewallRule {
                protocol: "tcp".to_string(),
                port: 443,
                source_ips: vec![format!("{}/32", public_ip)],
            },
        ];
        if let Err(e) = client.update_firewall_rules(rules).await {
            eprintln!("⚠️ Failed to update firewall rules: {}", e);
        } else {
            println!("✅ Firewall rules updated successfully.");
        }
    } else {
        println!("ℹ️ MY_PUBLIC_IP not set in .env, skipping firewall setup.");
    }

    // Wait for IP
    println!("⏳ Waiting for instance to be active and get an IP...");
    let mut instance_ip = None;
    for _ in 0..60 { // 10 minutes max
        tokio::time::sleep(std::time::Duration::from_secs(10)).await;
        if let Ok(instances_res) = client.get_instances().await {
            if let Some(inst) = instances_res.data.iter().find(|i| &i.id == instance_id) {
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
