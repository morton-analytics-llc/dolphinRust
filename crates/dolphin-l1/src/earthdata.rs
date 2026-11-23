//! NASA Earthdata Login (EDL) bearer token for ASF and CDDIS downloads.
//!
//! The token comes from the environment (or the repo `.env` loaded with
//! [`crate::load_env_file`]): `EARTHDATA_TOKEN` / `GP_EARTHDATA_TOKEN` while
//! its JWT `exp` is more than an hour away, otherwise the live token from
//! EDL's `find_or_create_token` with `EARTHDATA_USERNAME` /
//! `EARTHDATA_PASSWORD` (it returns the user's live token, or mints one, so
//! it never hits EDL's two-live-tokens cap). Same rule as the validation
//! helper `validation/edl_token.py` (#229). No token or password is logged.

use std::time::Duration;

use base64::Engine;
use reqwest::blocking::Client;
use serde::Deserialize;

use crate::error::{L1Error, Result};

const FIND_OR_CREATE_URL: &str = "https://urs.earthdata.nasa.gov/api/users/find_or_create_token";
/// A token this close to expiry is replaced.
const EXPIRY_MARGIN_S: i64 = 3600;

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

/// The JWT `exp` claim (epoch seconds), or `None` when it cannot be read.
fn token_expiry(token: &str) -> Option<i64> {
    let payload = token.split('.').nth(1)?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    claims.get("exp")?.as_f64().map(|e| e as i64)
}

fn usable(token: &str, now: i64) -> bool {
    !token.is_empty() && token_expiry(token).is_none_or(|exp| exp - EXPIRY_MARGIN_S > now)
}

/// A usable EDL bearer token.
///
/// # Errors
/// [`L1Error::MissingCredential`] when no usable token is set and the
/// username/password are missing; HTTP errors from EDL.
pub fn bearer_token() -> Result<String> {
    let pasted = ["GP_EARTHDATA_TOKEN", "EARTHDATA_TOKEN"]
        .iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.trim().is_empty()))
        .unwrap_or_default();
    if usable(pasted.trim(), chrono::Utc::now().timestamp()) {
        return Ok(pasted.trim().to_string());
    }
    let username = std::env::var("EARTHDATA_USERNAME")
        .map_err(|_| L1Error::MissingCredential("EARTHDATA_USERNAME"))?;
    let password = std::env::var("EARTHDATA_PASSWORD")
        .map_err(|_| L1Error::MissingCredential("EARTHDATA_PASSWORD"))?;
    let response = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?
        .post(FIND_OR_CREATE_URL)
        .basic_auth(username, Some(password))
        .send()?;
    if !response.status().is_success() {
        return Err(L1Error::Service {
            service: "Earthdata Login",
            status: response.status().as_u16(),
            url: FIND_OR_CREATE_URL.to_string(),
            body: String::new(),
        });
    }
    let token = response.json::<TokenResponse>()?.access_token;
    if token.is_empty() {
        return Err(L1Error::Product(
            "EDL find_or_create_token returned no access_token".into(),
        ));
    }
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(exp: i64) -> String {
        let engine = base64::engine::general_purpose::URL_SAFE_NO_PAD;
        format!(
            "{}.{}.sig",
            engine.encode(br#"{"alg":"none"}"#),
            engine.encode(format!(r#"{{"exp":{exp}}}"#))
        )
    }

    /// A token is used while its expiry is more than an hour out, and a
    /// token without a readable expiry is used as given.
    #[test]
    fn token_expiry_rule() {
        let now = 1_800_000_000;
        assert!(usable(&jwt(now + 7200), now));
        assert!(!usable(&jwt(now + 600), now));
        assert!(!usable(&jwt(now - 10), now));
        assert!(usable("opaque-token", now));
        assert!(!usable("", now));
    }
}
