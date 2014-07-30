//! A parser for Wavefront's `.mtl` file format, for storing information about
//! the material of which a 3D mesh is composed.
use std::result::{Result,Err};
pub use lex::ParseError;

/// Parses a wavefront `.mtl` file, returning either the successfully parsed
/// file, or an error. Support in this parser for the full file format is
/// best-effort and realistically I will only end up supporting the subset
/// of the file format which falls under the "shit I see exported from blender"
/// category.
pub fn parse(_input: &str) -> Result<(), ParseError> {
  Err(ParseError { line_number: 1, message: "Unimplemented.".into_string() })
}
