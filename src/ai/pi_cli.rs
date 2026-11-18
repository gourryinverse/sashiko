// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AI provider that shells out to the `pi` coding agent CLI
//! (<https://www.npmjs.com/package/@earendil-works/pi-coding-agent>).
//!
//! pi fronts many model providers behind one `--model provider/id` switch, so
//! this provider reaches any of them with pi's own configuration and
//! credentials. Sashiko drives it as a pure completion backend: every request
//! runs `pi --print --mode json` with all of pi's tools, skills, context files,
//! and session persistence turned off, in a fresh temporary directory, with a
//! neutral system prompt. Sashiko's ToolBox stays the only tool layer: tool
//! definitions travel in the prompt and tool calls come back as text, as for
//! the other CLI providers.
//!
//! pi's extensions stay loaded: a site's pi build may enforce policy through
//! them. `--no-tools` disables extension tools as well as built-in ones.
//!
//! The reply is the last assistant message of pi's JSON event stream
//! (`message_end`), with its token usage.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::debug;

use super::claude_cli::{build_prompt, log_and_redact_stderr, parse_inner_response};
use crate::ai::{
    AiProvider, AiRequest, AiResponse, AiUsage, ProviderCapabilities, cache_identity_with,
};
use crate::utils::redact_secret;

/// What pi is told it is: the request's own system prompt travels in the
/// prompt text, as for the other CLI providers.
const SYSTEM_PROMPT: &str = "You are a text completion backend. Follow the instructions \
in the user message exactly, and answer in the format it asks for.";

pub struct PiCliProvider {
    /// `provider/id` (pi's `--model`); empty for pi's default model.
    pub model: String,
    pub binary: String,
    /// pi's `--thinking` level; unset for the model default.
    pub thinking: Option<String>,
    pub context_window_size: usize,
    pub timeout_secs: u64,
}

/// Returns the argv passed to `pi` (the prompt goes on stdin). Extracted so
/// tests can assert that pi runs without tools.
pub fn build_pi_args(model: &str, thinking: Option<&str>) -> Vec<String> {
    let mut args: Vec<String> = [
        "--print",
        "--mode",
        "json",
        "--no-tools",
        "--no-skills",
        "--no-context-files",
        "--no-session",
        "--system-prompt",
        SYSTEM_PROMPT,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if !model.is_empty() {
        args.push("--model".to_string());
        args.push(model.to_string());
    }
    if let Some(thinking) = thinking {
        args.push("--thinking".to_string());
        args.push(thinking.to_string());
    }
    args
}

/// The text and usage of the last assistant message in pi's JSON event
/// stream, or why there is none.
pub fn parse_pi_output(raw: &str) -> Result<(String, Option<AiUsage>)> {
    let mut last: Option<Value> = None;
    for line in raw.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line).trim();
        if line.is_empty() {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if event["type"] == "message_end" && event["message"]["role"] == "assistant" {
            last = Some(event["message"].clone());
        }
    }
    let message = last.ok_or_else(|| anyhow::anyhow!("pi returned no assistant message"))?;
    if matches!(
        message["stopReason"].as_str(),
        Some("error") | Some("aborted")
    ) {
        let why = message["errorMessage"]
            .as_str()
            .unwrap_or("no reason given");
        anyhow::bail!("pi stopped with an error: {}", redact_secret(why));
    }
    let text = match &message["content"] {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|b| b["type"] == "text")
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    let u = &message["usage"];
    let usage = if u.is_object() {
        let input = u["input"].as_u64().unwrap_or(0) as usize;
        let output = u["output"].as_u64().unwrap_or(0) as usize;
        let cached = u["cacheRead"].as_u64().unwrap_or(0) as usize;
        Some(AiUsage {
            prompt_tokens: input,
            completion_tokens: output,
            total_tokens: u["totalTokens"]
                .as_u64()
                .map(|t| t as usize)
                .unwrap_or(input + output),
            cached_tokens: if cached > 0 { Some(cached) } else { None },
        })
    } else {
        None
    };
    Ok((text, usage))
}

#[async_trait]
impl AiProvider for PiCliProvider {
    async fn generate_content(&self, request: AiRequest) -> Result<AiResponse> {
        let prompt = build_prompt(&request);
        debug!("pi-cli prompt length: {} chars", prompt.len());

        let working_directory = tempfile::tempdir()
            .map_err(|e| anyhow::anyhow!("Failed to create pi CLI working directory: {}", e))?;
        let mut child = Command::new(&self.binary)
            .args(build_pi_args(&self.model, self.thinking.as_deref()))
            .current_dir(working_directory.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn pi CLI: {}. Is it installed?", e))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.flush().await?;
        }

        let output = timeout(
            Duration::from_secs(self.timeout_secs),
            child.wait_with_output(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("pi CLI timed out after {} seconds", self.timeout_secs))?
        .map_err(|e| anyhow::anyhow!("pi CLI wait error: {}", e))?;

        let stderr = log_and_redact_stderr("pi-cli", &output.stderr);
        let raw = String::from_utf8_lossy(&output.stdout);
        if !output.status.success() {
            anyhow::bail!("pi CLI exited with {}: {}", output.status, stderr);
        }
        let (text, usage) = parse_pi_output(&raw)?;
        parse_inner_response(&text, usage)
    }

    fn get_capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            model_name: if self.model.is_empty() {
                "pi-default".to_string()
            } else {
                self.model.clone()
            },
            context_window_size: self.context_window_size,
        }
    }

    fn cache_identity(&self) -> String {
        cache_identity_with(&self.model, &[("thinking", self.thinking.as_deref())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pi_runs_without_tools_skills_context_or_session() {
        let args = build_pi_args("anthropic/model-x", Some("high"));
        for flag in [
            "--print",
            "--no-tools",
            "--no-skills",
            "--no-context-files",
            "--no-session",
        ] {
            assert!(args.iter().any(|a| a == flag), "missing {flag}");
        }
        // Site policy lives in pi's extensions: they stay loaded.
        assert!(!args.iter().any(|a| a == "--no-extensions"));
        let at = args.iter().position(|a| a == "--mode").unwrap();
        assert_eq!(args[at + 1], "json");
        let at = args.iter().position(|a| a == "--model").unwrap();
        assert_eq!(args[at + 1], "anthropic/model-x");
        let at = args.iter().position(|a| a == "--thinking").unwrap();
        assert_eq!(args[at + 1], "high");
    }

    #[test]
    fn an_empty_model_leaves_pi_its_default() {
        let args = build_pi_args("", None);
        assert!(!args.iter().any(|a| a == "--model" || a == "--thinking"));
    }

    #[test]
    fn the_reply_is_the_last_assistant_message_with_its_usage() {
        let raw = concat!(
            r#"{"type":"session","version":3,"id":"x"}"#,
            "\n",
            r#"{"type":"message_end","message":{"role":"user","content":"hi"}}"#,
            "\n",
            r#"{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"Hel"}}"#,
            "\r\n",
            r#"{"type":"message_end","message":{"role":"assistant","stopReason":"stop","content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"Hello "},{"type":"text","text":"there"}],"usage":{"input":100,"output":7,"cacheRead":40,"cacheWrite":0,"totalTokens":107}}}"#,
            "\n",
            r#"{"type":"agent_settled","aborted":false}"#,
            "\n",
        );
        let (text, usage) = parse_pi_output(raw).unwrap();
        assert_eq!(text, "Hello there");
        let usage = usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 7);
        assert_eq!(usage.total_tokens, 107);
        assert_eq!(usage.cached_tokens, Some(40));
    }

    #[test]
    fn an_error_stop_is_an_error() {
        let raw = r#"{"type":"message_end","message":{"role":"assistant","stopReason":"error","errorMessage":"rate limited","content":[]}}"#;
        let err = parse_pi_output(raw).unwrap_err().to_string();
        assert!(err.contains("rate limited"), "{err}");
    }

    #[test]
    fn no_assistant_message_is_an_error() {
        let raw = r#"{"type":"session","version":3}"#;
        assert!(parse_pi_output(raw).is_err());
    }
}
