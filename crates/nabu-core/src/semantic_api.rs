//! Build-invariant semantic seam: the `Embedder` trait and embedding-unit
//! types, defined identically in both feature builds so that search and index
//! depend on a stable interface. The feature-gated implementation lives in the
//! `semantic` module.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub trait Embedder {
    fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f32>>>;
    fn embed_query(&self, query: &str) -> Result<Vec<f32>>;
    fn document_batch_size(&self) -> usize {
        16
    }
    fn intra_threads(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingUnitKind {
    UserText,
    AssistantText,
    ToolIntent,
    MetadataText,
}

impl EmbeddingUnitKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbeddingUnitKind::UserText => "user_text",
            EmbeddingUnitKind::AssistantText => "assistant_text",
            EmbeddingUnitKind::ToolIntent => "tool_intent",
            EmbeddingUnitKind::MetadataText => "metadata_text",
        }
    }
}

impl FromStr for EmbeddingUnitKind {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "user_text" => Ok(Self::UserText),
            "assistant_text" => Ok(Self::AssistantText),
            "tool_intent" => Ok(Self::ToolIntent),
            "metadata_text" => Ok(Self::MetadataText),
            _ => Err(Error::Validation(format!(
                "unsupported embedding unit kind: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingUnit {
    pub kind: EmbeddingUnitKind,
    pub unit_index: usize,
    pub text: String,
    pub text_hash: String,
}
