//! JavaScript error conversion, cooperative completion polling, and result objects.

use core::fmt::Display;

use js_sys::{Function, Promise, Reflect};
use minifield_engine_api::{CompletionPoll, InferenceCompletion};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

pub(super) fn to_string_vec(array: &js_sys::Array) -> Result<Vec<String>, JsValue> {
    let mut strings = Vec::with_capacity(array.length() as usize);
    for value in array.iter() {
        strings.push(
            value
                .as_string()
                .ok_or_else(|| JsValue::from_str("names and prompts must be strings"))?,
        );
    }
    Ok(strings)
}

pub(super) fn js_error(error: impl Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

pub(super) fn js_debug(error: impl core::fmt::Debug) -> JsValue {
    JsValue::from_str(&format!("{error:?}"))
}

/// Yield one macrotask so the browser can advance the WebGPU device timeline.
///
/// `map_async` and `on_submitted_work_done` resolve as JS promises only after
/// the event loop turns; a tight Rust poll loop would starve them forever.
/// The page provides `__minifieldYield` (a `MessageChannel` post, unclamped);
/// `setTimeout(0)` is the portable fallback.
pub(super) async fn browser_yield() {
    let global = js_sys::global();
    let helper = Reflect::get(&global, &JsValue::from_str("__minifieldYield"))
        .ok()
        .and_then(|value| value.dyn_into::<Function>().ok());
    let promise = helper
        .and_then(|helper| helper.call0(&JsValue::NULL).ok())
        .and_then(|value| value.dyn_into::<Promise>().ok())
        .unwrap_or_else(timeout_promise);
    let _ = JsFuture::from(promise).await;
}

fn timeout_promise() -> Promise {
    Promise::new(&mut |resolve, _reject| {
        let global = js_sys::global();
        let set_timeout = Reflect::get(&global, &JsValue::from_str("setTimeout"))
            .ok()
            .and_then(|value| value.dyn_into::<Function>().ok());
        match set_timeout {
            Some(set_timeout) => {
                let _ = set_timeout.call1(&global, &resolve);
            }
            None => {
                let _ = resolve.call0(&JsValue::NULL);
            }
        }
    })
}

pub(super) async fn pump<T: InferenceCompletion>(task: &mut T) -> Result<T::Output, JsValue> {
    loop {
        match task.poll_step() {
            CompletionPoll::Pending => browser_yield().await,
            CompletionPoll::Ready(result) => return result.map_err(js_error),
        }
    }
}

pub(super) async fn pump_future<T, E: Display>(
    task: impl std::future::Future<Output = Result<T, E>>,
) -> Result<T, JsValue> {
    let mut task = std::pin::pin!(task);
    loop {
        let poll = task
            .as_mut()
            .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        match poll {
            std::task::Poll::Ready(result) => return result.map_err(js_error),
            std::task::Poll::Pending => browser_yield().await,
        }
    }
}

pub(super) fn stats(text: &str, generated: usize, stopped: bool) -> Result<JsValue, JsValue> {
    let stats = js_sys::Object::new();
    Reflect::set(&stats, &JsValue::from_str("text"), &JsValue::from_str(text))?;
    Reflect::set(
        &stats,
        &JsValue::from_str("tokens"),
        &JsValue::from_f64(u32::try_from(generated).unwrap_or(u32::MAX).into()),
    )?;
    Reflect::set(
        &stats,
        &JsValue::from_str("stopped"),
        &JsValue::from_bool(stopped),
    )?;
    Ok(stats.into())
}
