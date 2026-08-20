// Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Victor Edge CLI — lightweight AI assistant for edge devices.
//!
//! # Usage
//!
//! ```bash
//! # Run a single task
//! victor-edge --task "Explain quicksort"
//!
//! # Interactive chat mode
//! victor-edge --interactive
//!
//! # Use a custom model
//! victor-edge --model llama3.2:3b --task "Write a haiku"
//!
//! # Connect to OpenAI
//! victor-edge --url https://api.openai.com --model gpt-4o --task "Hello"
//!
//! # Use a config file
//! victor-edge --config my-config.json --task "Analyze this"
//! ```

use clap::Parser;
use std::io::{self, BufRead, Write};
use victor_edge::agent::EdgeAgent;
use victor_edge::config::EdgeConfig;

/// Victor edge agent — lightweight AI assistant for resource-constrained devices.
#[derive(Parser)]
#[command(
    name = "victor-edge",
    about = "Victor edge agent — lightweight AI assistant",
    version
)]
struct Cli {
    /// Task to execute (runs once and exits).
    #[arg(short, long)]
    task: Option<String>,

    /// Path to a JSON config file.
    #[arg(short, long, default_value = "victor-edge.json")]
    config: String,

    /// Model to use (overrides config file).
    #[arg(short, long)]
    model: Option<String>,

    /// Provider URL (overrides config file).
    #[arg(long)]
    url: Option<String>,

    /// Interactive chat mode (REPL).
    #[arg(short, long)]
    interactive: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (respects RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    // Load or create config
    let mut config = if std::path::Path::new(&cli.config).exists() {
        EdgeConfig::from_file(&cli.config)?
    } else {
        EdgeConfig::default()
    };

    // Apply CLI overrides
    if let Some(model) = cli.model {
        config.provider.model = model;
    }
    if let Some(url) = cli.url {
        config.provider.base_url = url;
    }

    let mut agent = EdgeAgent::new(config.to_agent_config());

    if let Some(task) = cli.task {
        // Single-task mode
        let result = agent.run(&task).await?;
        println!("{result}");
    } else if cli.interactive {
        // Interactive REPL mode
        println!("Victor Edge Agent (type 'exit' or 'quit' to stop)");
        println!("---");
        let stdin = io::stdin();
        loop {
            print!("> ");
            io::stdout().flush()?;
            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            let line = line.trim();
            if line.is_empty() || line == "exit" || line == "quit" {
                break;
            }
            match agent.chat(line).await {
                Ok(response) => println!("{response}"),
                Err(e) => eprintln!("Error: {e}"),
            }
        }
    } else {
        println!("Usage: victor-edge --task 'your task' or victor-edge --interactive");
        println!("Run victor-edge --help for all options.");
    }

    Ok(())
}
