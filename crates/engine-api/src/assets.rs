//! Bounded asset access and exact imported tensor inventories.

use std::collections::{BTreeMap, BTreeSet};

use crate::{ByteRange, CompletionPoll, DType, ExecutorError, InferenceCompletion, Result, Shape};

/// Immutable bytes delivered from an asset provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetBytes {
    bytes: Vec<u8>,
}

impl AssetBytes {
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Filesystem-free input boundary. Native files and WASM asset buffers adapt to this trait.
pub trait AssetProvider {
    type Read: InferenceCompletion<Output = AssetBytes>;

    fn read_range(&mut self, range: ByteRange) -> Result<Self::Read>;
}

/// Asset limits checked before any loader allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssetLimits {
    pub max_asset_bytes: u64,
    pub max_tensor_bytes: u64,
    pub max_tensors: usize,
}

/// Tensor record supplied by a format-specific importer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRecord {
    pub name: String,
    pub dtype: DType,
    pub shape: Shape,
    pub bytes: ByteRange,
}

impl TensorRecord {
    pub fn validate(&self, asset_bytes: u64, limits: AssetLimits) -> Result<()> {
        if self.name.is_empty() {
            return Err(ExecutorError::InvalidArgument("tensor name is empty"));
        }
        self.bytes.validate_within(asset_bytes)?;
        let expected_bytes = self
            .shape
            .element_count()?
            .checked_mul(self.dtype.byte_width())
            .ok_or(ExecutorError::Overflow("tensor byte length overflows u64"))?;
        if expected_bytes != self.bytes.len {
            return Err(ExecutorError::InvalidLayout(
                "tensor range differs from shape and dtype byte length",
            ));
        }
        if self.bytes.len > limits.max_tensor_bytes {
            return Err(ExecutorError::ResourceLimit(
                "tensor exceeds configured byte limit",
            ));
        }
        Ok(())
    }
}

/// Explicit model role requirement. Ties use another role and must name the same source tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRequirement {
    pub role: String,
    pub tensor_name: String,
    pub dtype: DType,
    pub shape: Shape,
    pub tied_to_role: Option<String>,
}

/// Portable imported-asset manifest. Parsing of concrete formats belongs in executor core later.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssetManifest {
    pub config_name: String,
    pub asset_bytes: u64,
    pub tensors: Vec<TensorRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorBinding {
    pub role: String,
    pub tensor_index: usize,
}

/// Maximum requirement roles accepted by this foundation before a concrete model importer
/// applies its tighter, model-specific inventory.
pub const MAX_TENSOR_REQUIREMENTS: usize = 4096;

/// Validate an exact config and physical tensor inventory.
///
/// Requirements are an internal model declaration. A canonical physical role must appear before
/// each direct alias. Alias chains, forward references, self references, and duplicate canonical
/// roles are invalid. Each nonempty tensor byte range must be disjoint from every other physical
/// tensor record; tied roles share one record rather than overlapping records.
#[allow(clippy::too_many_lines)]
pub fn validate_asset_manifest(
    manifest: &AssetManifest,
    expected_config_name: &str,
    requirements: &[TensorRequirement],
    limits: AssetLimits,
) -> Result<Vec<TensorBinding>> {
    if manifest.config_name != expected_config_name {
        return Err(ExecutorError::InvalidArgument(
            "unexpected model config name",
        ));
    }
    if manifest.asset_bytes > limits.max_asset_bytes {
        return Err(ExecutorError::ResourceLimit(
            "asset exceeds configured byte limit",
        ));
    }
    if manifest.tensors.len() > limits.max_tensors {
        return Err(ExecutorError::ResourceLimit(
            "tensor count exceeds configured limit",
        ));
    }
    if requirements.len() > MAX_TENSOR_REQUIREMENTS {
        return Err(ExecutorError::ResourceLimit(
            "tensor requirement count exceeds portable limit",
        ));
    }

    let mut requirement_roles = BTreeMap::new();
    let mut canonical_roles_by_tensor = BTreeMap::new();
    let mut required_tensor_names = BTreeSet::new();
    for (index, requirement) in requirements.iter().enumerate() {
        if requirement.role.is_empty() || requirement.tensor_name.is_empty() {
            return Err(ExecutorError::InvalidArgument(
                "tensor requirement name is empty",
            ));
        }
        if requirement_roles
            .insert(requirement.role.as_str(), index)
            .is_some()
        {
            return Err(ExecutorError::DuplicateName);
        }
        match requirement.tied_to_role.as_deref() {
            None => {
                if canonical_roles_by_tensor
                    .insert(requirement.tensor_name.as_str(), requirement.role.as_str())
                    .is_some()
                {
                    return Err(ExecutorError::InvalidTie);
                }
            }
            Some(tied_to_role) => {
                if tied_to_role == requirement.role {
                    return Err(ExecutorError::InvalidTie);
                }
                let Some(target_index) = requirement_roles.get(tied_to_role).copied() else {
                    return Err(ExecutorError::InvalidTie);
                };
                let target = &requirements[target_index];
                if target.tied_to_role.is_some()
                    || target.tensor_name != requirement.tensor_name
                    || target.dtype != requirement.dtype
                    || target.shape != requirement.shape
                    || canonical_roles_by_tensor.get(requirement.tensor_name.as_str())
                        != Some(&target.role.as_str())
                {
                    return Err(ExecutorError::InvalidTie);
                }
            }
        }
        required_tensor_names.insert(requirement.tensor_name.as_str());
    }

    let mut tensor_indexes = BTreeMap::new();
    for (index, tensor) in manifest.tensors.iter().enumerate() {
        tensor.validate(manifest.asset_bytes, limits)?;
        if !required_tensor_names.contains(tensor.name.as_str()) {
            return Err(ExecutorError::UnexpectedTensor);
        }
        if tensor_indexes.insert(tensor.name.as_str(), index).is_some() {
            return Err(ExecutorError::DuplicateName);
        }
    }
    if manifest.tensors.len() != required_tensor_names.len() {
        return Err(ExecutorError::UnexpectedTensor);
    }
    for tensor_name in &required_tensor_names {
        if !tensor_indexes.contains_key(tensor_name) {
            return Err(ExecutorError::MissingRequiredTensor);
        }
    }
    for (left_index, left) in manifest.tensors.iter().enumerate() {
        if left.bytes.len == 0 {
            continue;
        }
        let left_end = left.bytes.end()?;
        for right in manifest.tensors.iter().skip(left_index + 1) {
            if right.bytes.len == 0 {
                continue;
            }
            let right_end = right.bytes.end()?;
            if left.bytes.offset < right_end && right.bytes.offset < left_end {
                return Err(ExecutorError::InvalidLayout(
                    "physical tensor byte ranges overlap",
                ));
            }
        }
    }

    let mut bindings = Vec::with_capacity(requirements.len());
    for requirement in requirements {
        let tensor_index = tensor_indexes
            .get(requirement.tensor_name.as_str())
            .copied()
            .ok_or(ExecutorError::MissingRequiredTensor)?;
        let tensor = &manifest.tensors[tensor_index];
        if tensor.dtype != requirement.dtype || tensor.shape != requirement.shape {
            return Err(ExecutorError::InvalidShape(
                "tensor does not match required shape or dtype",
            ));
        }
        bindings.push(TensorBinding {
            role: requirement.role.clone(),
            tensor_index,
        });
    }
    Ok(bindings)
}

/// In-memory portable asset provider suitable for embedded and WASM callers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryAssetProvider {
    bytes: Vec<u8>,
    max_read_bytes: u64,
}

impl MemoryAssetProvider {
    #[must_use]
    pub fn new(bytes: Vec<u8>, max_read_bytes: u64) -> Self {
        Self {
            bytes,
            max_read_bytes,
        }
    }
}

/// Immediate completion returned by the in-memory asset provider.
#[derive(Debug)]
pub struct MemoryAssetRead {
    result: Option<Result<AssetBytes>>,
}

impl AssetProvider for MemoryAssetProvider {
    type Read = MemoryAssetRead;

    fn read_range(&mut self, range: ByteRange) -> Result<Self::Read> {
        range.validate_within(
            u64::try_from(self.bytes.len())
                .map_err(|_| ExecutorError::Overflow("asset length exceeds u64"))?,
        )?;
        if range.len > self.max_read_bytes {
            return Err(ExecutorError::ResourceLimit(
                "asset read exceeds configured byte limit",
            ));
        }
        let start = usize::try_from(range.offset)
            .map_err(|_| ExecutorError::Overflow("asset offset exceeds usize"))?;
        let end = usize::try_from(range.end()?)
            .map_err(|_| ExecutorError::Overflow("asset end exceeds usize"))?;
        Ok(MemoryAssetRead {
            result: Some(Ok(AssetBytes::new(self.bytes[start..end].to_vec()))),
        })
    }
}

impl InferenceCompletion for MemoryAssetRead {
    type Output = AssetBytes;

    fn poll_step(&mut self) -> CompletionPoll<Self::Output> {
        match self.result.take() {
            Some(result) => CompletionPoll::Ready(result),
            None => CompletionPoll::Ready(Err(ExecutorError::CompletionConsumed)),
        }
    }

    fn cancel(&mut self) -> Result<()> {
        if self.result.is_some() {
            self.result = Some(Err(ExecutorError::Cancelled));
            Ok(())
        } else {
            Err(ExecutorError::CompletionConsumed)
        }
    }
}
