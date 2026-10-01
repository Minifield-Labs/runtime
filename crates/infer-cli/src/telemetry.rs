//! Native host delivery. The completed result has already been written and flushed.
use minifield_runtime_telemetry::ENDPOINT;
use serde_json::{Value, json};
use std::{sync::mpsc, time::Duration};

const TIMEOUT: Duration = Duration::from_secs(5);

fn label(value: Option<String>) -> Option<String> {
    value.filter(|s| {
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._:/@+-".contains(&c))
    })
}

fn configuration(mut get: impl FnMut(&str) -> Option<String>) -> Option<(String, Value)> {
    if get("MINIFIELD_TELEMETRY")
        .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "0" | "false" | "off"))
    {
        return None;
    }
    let endpoint = get("MINIFIELD_TELEMETRY_ENDPOINT").unwrap_or_else(|| ENDPOINT.into());
    let uri: ureq::http::Uri = endpoint.parse().ok()?;
    let scheme = uri.scheme_str()?;
    let host = uri.host()?;
    if uri.authority()?.as_str().contains('@')
        || uri.query().is_some()
        || endpoint.contains('#')
        || (scheme != "https"
            && !(scheme == "http" && matches!(host, "127.0.0.1" | "localhost" | "[::1]")))
    {
        return None;
    }
    let environment = get("MINIFIELD_ENVIRONMENT")
        .filter(|v| {
            matches!(
                v.as_str(),
                "production" | "preview" | "development" | "test"
            )
        })
        .unwrap_or_else(|| "production".into());
    let origin = json!({
        "integration_id": label(get("MINIFIELD_INTEGRATION_ID")).unwrap_or_else(|| "minifield-infer".into()),
        "product_id": label(get("MINIFIELD_PRODUCT_ID")), "website_origin": null,
        "application_id": label(get("MINIFIELD_APPLICATION_ID")),
        "application_version": label(get("MINIFIELD_APPLICATION_VERSION")), "environment": environment
    });
    Some((endpoint, origin))
}

pub fn report(mut record: Value) {
    let Some((endpoint, origin)) = configuration(|key| std::env::var(key).ok()) else {
        return;
    };
    record["origin"] = origin;
    record["hardware"] = json!({ "kind": "cpu", "vendor": null, "architecture": match std::env::consts::ARCH {
        "aarch64" | "arm" => "arm", "x86_64" => "x64", other => other,
    }});
    record["platform"] = json!({ "host": "native", "os": { "family": std::env::consts::OS, "major_version": null }, "browser": null });
    let body = record.to_string() + "\n";
    let (tx, rx) = mpsc::sync_channel(1);
    let spawned = std::thread::Builder::new()
        .name("minifield-telemetry".into())
        .spawn(move || {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_global(Some(TIMEOUT))
                .max_redirects(0)
                .build()
                .into();
            let _ = agent
                .post(&endpoint)
                .header("Content-Type", "application/stream+json")
                .send(body);
            let _ = tx.send(());
        });
    if spawned.is_ok() {
        let _ = rx.recv_timeout(TIMEOUT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disable_and_endpoint_validation_are_independent_of_inference() {
        assert!(configuration(|k| (k == "MINIFIELD_TELEMETRY").then(|| "0".into())).is_none());
        for invalid in [
            "http://example.com/",
            "https://user:secret@example.com/",
            "https://example.com/?secret=value",
        ] {
            assert!(
                configuration(|k| (k == "MINIFIELD_TELEMETRY_ENDPOINT").then(|| invalid.into()))
                    .is_none()
            );
        }
        assert!(configuration(|_| None).is_some());
        assert!(
            configuration(|k| (k == "MINIFIELD_TELEMETRY_ENDPOINT")
                .then(|| "http://127.0.0.1:8000/insert/jsonline".into()))
            .is_some()
        );
    }
}
